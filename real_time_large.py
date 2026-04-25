import sys
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque

import numpy as np
import sounddevice as sd
import webrtcvad
import matplotlib
matplotlib.use("TkAgg")  # Tkinter backend
import matplotlib.pyplot as plt

from faster_whisper import WhisperModel

# ===================== User Config =====================
SAMPLE_RATE         = 16000
FRAME_MS            = 20
CHUNK_SECONDS       = 1.0      # 짧게 말해도 반응하도록 최소 청크를 1.0s로
OVERLAP_SECONDS     = 0.30
SILENCE_TIMEOUT     = 0.28     # 말 멈추면 빠르게 플러시
MAX_ACTIVE_SECONDS  = 10.0
LANG                = "ko"     # 인식 언어 (Whisper용, 콘솔 출력만)

MODEL_SOURCE        = "local"
LOCAL_MODEL_DIR     = r"/\whisper_large"
HUB_MODEL_NAME      = "Systran/faster-whisper-large-v3"
HF_CACHE_DIR        = None
LOCAL_ONLY          = True

PREFER_FLOAT16      = True
BEAM_SIZE           = 2
PATIENCE = 1.0   # ← 0.12 에서 1.0으로
INITIAL_GUIDE       = "한국어로 자연스럽게 띄어쓰고 구두점을 사용해 주세요."
CONTEXT_CHARS       = 160

# ===================== Derived =====================
FRAME_SAMPLES        = SAMPLE_RATE * FRAME_MS // 1000
OVERLAP_SAMPLES      = int(OVERLAP_SECONDS * SAMPLE_RATE)
MIN_SAMPLES          = int(CHUNK_SECONDS * SAMPLE_RATE)

# 아주 짧은 발화도 내보내기 위한 하한(0.6초)
MIN_FLUSH_SECONDS    = 0.60
MIN_FLUSH_SAMPLES    = int(MIN_FLUSH_SECONDS * SAMPLE_RATE)

MAX_ACTIVE_SAMPLES   = int(MAX_ACTIVE_SECONDS * SAMPLE_RATE)

audio_q   = queue.Queue(maxsize=64)
decode_q  = queue.Queue(maxsize=12)   # 버스트 발화 대비
stop_flag = False
prev_context = ""

class VisualState:
    def __init__(self, wave_seconds=6.0):
        self.lock = threading.Lock()
        self.wave_buf = deque(maxlen=int(SAMPLE_RATE * wave_seconds))
        self.rms = 0.0
        self.is_speech = False

vis = VisualState(wave_seconds=6.0)

# --------------------------------------
# Model
# --------------------------------------
def pick_device_and_compute():
    device = "cpu"; compute_type = "int8"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16" if PREFER_FLOAT16 else "int8_float16"
    except:
        pass
    return device, compute_type

def load_model():
    device, compute_type = pick_device_and_compute()
    print(f"[INFO] Device={device}, compute_type={compute_type}")
    src = LOCAL_MODEL_DIR if MODEL_SOURCE == "local" else HUB_MODEL_NAME
    kwargs = dict(device=device, compute_type=compute_type)
    if HF_CACHE_DIR: kwargs["cache_directory"] = HF_CACHE_DIR
    if LOCAL_ONLY:   kwargs["local_files_only"] = True
    model = WhisperModel(src, **kwargs)
    # warm-up (가볍게)
    _warm = np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.float32)
    _ = list(model.transcribe(
        _warm, language=LANG, beam_size=1,
        vad_filter=False, condition_on_previous_text=False,
        word_timestamps=False
    ))
    return model

# --------------------------------------
# Audio callback
# --------------------------------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    try:
        audio_q.put_nowait(indata[:, 0].copy())
    except queue.Full:
        pass

# --------------------------------------
# Decoder worker (예외 로그 + 방어 처리 + RTT 측정)
# --------------------------------------
def decode_worker(model: WhisperModel):
    local_patience = max(1.0, PATIENCE)  # 안전장치
    import traceback
    global prev_context
    print("[DEBUG] decode_worker started", flush=True)
    while not stop_flag:
        try:
            item = decode_q.get(timeout=0.1)
        except queue.Empty:
            continue

        # 큐 아이템 언패킹 (segment 또는 (segment, enqueue_time))
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (int, float)):
            segment, t0 = item
        else:
            segment, t0 = item, None

        try:
            # ---- 방어적 변환: C-연속 float32, NaN/Inf 제거 ----
            seg = np.asarray(segment, dtype=np.float32)
            if not seg.flags['C_CONTIGUOUS']:
                seg = np.ascontiguousarray(seg)
            if np.any(~np.isfinite(seg)):
                seg = np.nan_to_num(seg, copy=False)

            # 너무 짧거나 거의 무음이면 스킵 (0.3s, 아주 낮은 RMS)
            if seg.size < int(0.3 * SAMPLE_RATE) or np.sqrt(np.mean(seg**2)) < 1e-4:
                decode_q.task_done()
                continue

            # ---- 추론 ----
            segments, info = model.transcribe(
                seg,
                language=LANG,
                task="transcribe",
                beam_size=BEAM_SIZE,
                patience=local_patience,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=True,   # 문맥 유지
                initial_prompt=(prev_context + " " + INITIAL_GUIDE).strip() if INITIAL_GUIDE else None,
                word_timestamps=False,
            )

            out = []
            for s in segments:
                out.append(s.text)
                # 세그먼트별 디버그 보고 싶으면 아래 주석 해제
                # print(f"[DBG seg] {s.start:.2f}-{s.end:.2f} | {s.text!r}")

            text = "".join(out).strip()

            if text:
                if t0 is not None:
                    print(f"[DEBUG] rtt { (time.time()-t0)*1000:.0f} ms", flush=True)
                print(f"[FINAL] {text}", flush=True)
                joined = (prev_context + " " + text).strip()
                if CONTEXT_CHARS > 0:
                    prev_context = joined[-CONTEXT_CHARS:]
                else:
                    prev_context = ""
            else:
                # 빈 결과면 이유 로그
                lang = getattr(info, 'language', '?')
                lprob = getattr(info, 'language_probability', '?')
                print(f"[WARN] empty text | dur={seg.size/SAMPLE_RATE:.2f}s | lang={lang} | prob={lprob}", flush=True)

        except Exception:
            print("[ERROR] decode_worker exception:", flush=True)
            traceback.print_exc()
        finally:
            decode_q.task_done()

# --------------------------------------
# VAD streaming
# --------------------------------------
def vad_streamer():
    # 과격도 1(보통). 과도하게 높이면 짧은 발화가 누락될 수 있음.
    vad = webrtcvad.Vad(1)
    pending = np.zeros(0, dtype=np.float32)
    active  = np.zeros(0, dtype=np.float32)
    tail    = np.zeros(0, dtype=np.float32)

    last_speech_ts = 0.0
    silence_armed  = False

    def is_speech(frame):
        f = np.clip(frame * 32768.0, -32768.0, 32767.0).astype(np.int16)
        return vad.is_speech(f.tobytes(), SAMPLE_RATE)

    while not stop_flag:
        try:
            block = audio_q.get(timeout=0.1)
            with vis.lock:
                vis.wave_buf.extend(block.tolist())
                vis.rms = float(np.sqrt(np.mean(np.square(block))) if block.size else 0.0)
        except queue.Empty:
            block = None

        if block is not None:
            pending = np.concatenate((pending, block)) if pending.size else block

        while pending.size >= FRAME_SAMPLES:
            frame = pending[:FRAME_SAMPLES]
            pending = pending[FRAME_SAMPLES:]

            speech_flag = is_speech(frame)
            now = time.time()

            if speech_flag:
                with vis.lock:
                    vis.is_speech = True
                silence_armed = True
                last_speech_ts = now
                active = frame if active.size == 0 else np.concatenate((active, frame))

                if active.size > MAX_ACTIVE_SAMPLES:
                    segment = np.concatenate((tail, active)) if tail.size else active.copy()
                    enqueue_time = time.time()
                    try:
                        decode_q.put_nowait((segment, enqueue_time))
                        print(f"[DEBUG] push(max) {segment.size/SAMPLE_RATE:.2f}s", flush=True)
                    except queue.Full:
                        pass
                    tail = segment[-OVERLAP_SAMPLES:] if segment.size >= OVERLAP_SAMPLES else segment.copy()
                    active = np.zeros(0, dtype=np.float32)
                    silence_armed = False
                    with vis.lock:
                        vis.is_speech = False
            else:
                if silence_armed and (now - last_speech_ts) >= SILENCE_TIMEOUT:
                    # 핵심: 너무 짧아도 흘려보내지 않도록 하한을 0.6s로 완화
                    if active.size >= MIN_FLUSH_SAMPLES:
                        segment = np.concatenate((tail, active)) if tail.size else active.copy()
                        enqueue_time = time.time()
                        try:
                            decode_q.put_nowait((segment, enqueue_time))
                            print(f"[DEBUG] push(silence) {segment.size/SAMPLE_RATE:.2f}s", flush=True)
                        except queue.Full:
                            pass
                        tail = segment[-OVERLAP_SAMPLES:] if segment.size >= OVERLAP_SAMPLES else segment.copy()
                    active = np.zeros(0, dtype=np.float32)
                    silence_armed = False
                    with vis.lock:
                        vis.is_speech = False

# --------------------------------------
# Visualization
# --------------------------------------
def start_visualization():
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)
    ax_wave = fig.add_subplot(gs[0, 0])
    ax_lvl  = fig.add_subplot(gs[1, 0])

    x_wave = np.arange(vis.wave_buf.maxlen) / SAMPLE_RATE
    y_wave = np.zeros(vis.wave_buf.maxlen)
    (line_wave,) = ax_wave.plot(x_wave, y_wave, lw=1.0)
    ax_wave.set_title("Real-time Waveform")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_ylim(-1.0, 1.0)
    ax_wave.set_xlim(x_wave[0], x_wave[-1])
    ax_wave.grid(True, alpha=0.3)

    lvl_bar = ax_lvl.bar([0], [0.0])[0]
    ax_lvl.set_ylim(0, 0.3)
    ax_lvl.set_xlim(-0.5, 0.5)
    ax_lvl.set_xticks([])
    ax_lvl.set_ylabel("RMS")
    txt_status = ax_lvl.text(0.02, 0.9, "STATE: ...", transform=ax_lvl.transAxes, va='top')

    def update_once():
        with vis.lock:
            buf_list = list(vis.wave_buf)
            rms = vis.rms
            is_sp = vis.is_speech

        N = len(buf_list)
        if N < vis.wave_buf.maxlen:
            y = np.zeros(vis.wave_buf.maxlen)
            if N > 0:
                y[-N:] = np.array(buf_list)
        else:
            y = np.array(buf_list[-vis.wave_buf.maxlen:])

        line_wave.set_ydata(y)

        lvl_bar.set_height(rms)
        txt_status.set_text(f"STATE: {'SPEECH' if is_sp else 'SILENCE'} | RMS: {rms:.3f}")
        lvl_bar.set_color("#2ca02c" if is_sp else "#1f77b4")

    timer = fig.canvas.new_timer(interval=33)
    def on_timer(*args, **kwargs):
        update_once()
        fig.canvas.draw_idle()
    timer.add_callback(on_timer)
    timer.start()
    plt.show()

# --------------------------------------
# Main
# --------------------------------------
def main():
    global stop_flag
    print("마이크에 말을 해주세요.")

    sd.default.latency = ('low', 'low')
    model = load_model()
    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(decode_worker, model)

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=FRAME_SAMPLES, callback=audio_callback)
    stream.start()

    t_vad = threading.Thread(target=vad_streamer, daemon=True)
    t_vad.start()

    try:
        start_visualization()
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag = True
        time.sleep(0.2)
        try:
            stream.stop(); stream.close()
        except:
            pass  # passx 오타 방지
        executor.shutdown(wait=False)
        print("\n종료중.")

if __name__ == "__main__":
    main()
