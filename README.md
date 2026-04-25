# STT 
## 사용모델 - OpenAi model whisper

---

---

- whisper?
    
    OpenAI의 **Whisper** 모델은 자동 음성 인식(ASR, Automatic Speech Recognition)과 다국어 음성 처리에 특화된 **딥러닝 기반 음성-텍스트 변환 모델**
    
    ---
    
    - **Transformer 기반 인코더-디코더 구조**
    - **68만 시간 이상**의 다국어 음성 데이터(자막 포함)로 학습
    - 모델 크기 → 모델이 작아질수록 빠르지만 정확도가 낮아짐…
        - `tiny` (39M)
        - `base` (74M)
        - `small` (244M)
        - `medium` (769M)
        - `large` (1550M)
    - Why?
        
        ---
        
        - 장점
            
            ---
            
            - 범용성 (Multilingual + Multi-task)
                - 다국어 지원, **98개 이상 언어**를 포함해 자연스럽게 인식
                - 번역까지 내장 → 음성을 바로 **영어로 번역**
            
            ---
            
            - 오픈소스 + 로컬 실행 가능
                - 클라우드 의존 없이 완전 오프라인 실행
                    - Google Speech API, Azure Speech 등은 클라우드 사용이 기본
                    - Whisper는 공개 모델을 내려받아 GPU/CPU에서 오프라인으로 실행가능
                - 상용 API 대비 무료
            
            ---
            
            - 잡음 강인성 (Noise Robustness)
                - 68만 시간 이상의 실제 유튜브/자막 데이터로 학습되어 **소음, 억양, 발화 스타일 변화**에 대한 적응력이 높다
            
            ---
            
            - 정확도 + 문맥 이해력
                - Transformer 기반 문맥 처리, 길이 제한 없음
            
            ---
            
            - 개발 친화성 & 생태계
                - Python API 간편
        
        ---
        
        - 타 API와 비교
            
            
            | 특성 | Whisper | Google Speech API | Vosk / DeepSpeech |
            | --- | --- | --- | --- |
            | 다국어 | ✅ 매우 넓음 | 제한적 | 언어별 모델 필요 |
            | 번역 | ✅ 내장 | ❌ 별도 API 필요 | ❌ |
            | 오프라인 사용 | ✅ 가능 | ❌ | ✅ 가능 |
            | 잡음 강인성 | ✅ 높음 | 중간 | 낮음 |
            | 설치/사용 편의성 | ✅ 쉬움 | API Key 필요 | 모델 빌드 필요 |
            | 실시간 최적화 | ✅ 가능(faster-whisper) | ✅ | 부분적으로 지원 |

---

---

- What
    - **faster-whisper (CTranslate2 기반, GPU/CPU 최적화)**
    - ~~WebRTC + Whisper API (OpenAI Realtime)~~
    - ~~VAD(Voice Activity Detection)와 조합해 말할 때만 캡처 → 더 자연스러운 실시간 대화 구현~~

---

---

- How
    - faster-whisper

---

---

- Development Step
    - 1. 음성녹음을 통한 stt
        
        ```python
        import whisper
        from IPython.display import Audio, display
        
        audio_path = "D:\개인\이것저것\갠프\음성인식_whisper\유상록 교수님 파일\example.wav"
        
        display(Audio(audio_path))
        
        a = input("모델 선택(tiny, base, small, medium, large)")
        model = whisper.load_model(a)
        
        result = whisper.transcribe(model)
        print(result["text"])
        ```
        
    - 2. 실시간 stt
        
    - 3. 서버모델 → 로컬로 전환
        
    - 4. 정확도 향상 & 속도 증강
        
    - 5. 마이크 입력 시각화(입력이 되는건지 안되는건지 확인하기 위해)           

---

---

- Problem
    - 직접 겪은 문제
        
        whisper모델은 공식적으로 3.13.x 지원하지않음  
        
        →  3.10. x ~ 3.11.x 에서 사용
        
        ---
        
        특정 단어를 잘 알아먹지 못함 ex) 우로 → 위로, 오로
        
        →  테스트를 더 해봐야 할 것 같음 한번씩 우로 → 우측으로 등의 잘못 받아들이는 경우가 있긴 하나 결과는 좋으니 좋은게 좋은거지 라는 생각이 있음….
        
        ---
        
        tiny, base 모델은 쓰지말자 속도는 매우 빠르나 정확도가 굉장히 떨어짐
        
        large모델이 성능상 좋긴 하다만 매우 오래걸림  
        
        ![image.png](image%201.png)
        
              → 실시간 운용에서는 이렇게까지 걸리지 않음
        
        → GPU사용시 속도 상승
        
        ---
        
        실시간 작동시 정확도가 조금 떨어지는 경향이 있음 
        
        → 이유는 모르겠으나 파이참이나 vs코드가 아닌 cmd로 실행시 정확도가 올라감
        
    
    ---
    
    - 기타 문제
        
        ---
        
        - **실시간 지연 (Latency)**
            - CPU만 사용할 경우 **실시간 처리 속도가 느림** (특히 medium/large 모델)
            - GPU 사용, 모델 크기 축소(tiny/base), INT8 변환 등으로 보완 필요
        
        ---
        
        - **모델 크기와 자원 요구**
            - large 모델 1.5B 파라미터 → GPU 메모리 최소 8GB 이상 권장
            - 임베디드/저사양 장치에는 부담
        
        ---
        
        - **언어별 편차**
            - 영어에 최적화 → 비주류 언어는 WER이 상대적으로 높을 수 있음
            - 한국어도 좋지만 Google STT 등 클라우드가 일부 상황에서 더 나을 수 있음
                
                
                | 상황 | Whisper | Google STT 등 클라우드 |
                | --- | --- | --- |
                | 저지연 실시간 | ❌ (딜레이 있음) | ✅ 빠른 반응 |
                | 저사양/모바일 | ❌ 무겁고 느림 | ✅ 클라우드 처리 |
                | 방언/전문 용어 | △ 일반적 | ✅ 지역/분야 특화 |
                | 초장시간 오디오 | △ 리소스 부담 | ✅ 안정적 처리 |
                | 최신 용어 반영 | ❌ 업데이트 드묾 | ✅ 지속 업데이트 |
        
        ---
        
        - **발화자 분리(Speaker Diarization) 없음**
            - 기본 모델은 단일 음성 전사 전용 → 화자 분리 기능 필요하면 pyannote.audio 등과 연동해야 함
        
        ---
        
        - **모델 업데이트 빈도 낮음**
            - OpenAI가 Whisper를 자주 개선하지 않음 (커뮤니티 포크/최적화 버전 사용 권장)
        
        ---
        
        - **네트워크로 직접 스트리밍 지원 X**
            - Realtime API 이전의 오픈소스 Whisper는 WebRTC/Socket 같은 실시간 통신은 직접 구현해야 함
        
        ---
        
        - Ubuntu(Linux)환경에서 사용 가능한가?
            - 가능은 하나 Ubuntu환경에서 마이크 인식을 위한 드라이버를 따로 설치하는등의 과정이 필요 → 번거롭다.
                - 윈도우 환경에서 stt후 텍스트만 네트워크를 통해 넘긴다.
                - stt이후 모든 과정까지 전부 윈도우로 해결한 뒤 Ubuntu로 넘긴다.

---

---

- Ubuntu
    - 현재 문제점
        1. 우분투환경에서 터미널 실행이 안됨
        2. 실행중 어떤 마이크로 입력받는지 정확히 모름(제일 문제)
            1. 마이크 입력이 제대로 되는지 모르기 때문에 제대로 테스트를 해볼수 없음
                
                → 어떤 시점에 입력이 들어가서 몇초 후에 출력이 나오는지 확인 불가
                
        3. gpu연결문제 - 우분투가 gpu인식을 못함

---

---

- About
    - 성능향상 위주
        
        https://github.com/SYSTRAN/faster-whisper
        

---

---

- Task
  - 잡음 제거에 가장 탁월한 알고리즘 조사하기
    - 잡음 제거에 탁월한 알고리즘
        
        
        | 순위 | 알고리즘 | 실시간성 | 잡음 제거 성능 | CPU/GPU 부하 | 비고 |
        | --- | --- | --- | --- | --- | --- |
        | 1 | **RNNoise** | 최고 (10–20 ms) | 중~고 | 보통 | Whisper 파이프라인 최적 |
        | 2 | **WebRTC NS** | 높음 | 중 | 낮음 | webrtcvad와 바로 연동 가능 |
        | 3 | **Spectral Gating (noisereduce)** | 낮음 | 높음 | 높음 | 녹음 파일 처리용 |

---

---

- Parameter
    - 전체 파라미터
        
        
        | 오디오 | `FRAME_MS`  | `CHUNK_SECONDS`* | `OVERLAP_SECONDS`* | `SILENCE_TIMEOUT`* |
        | --- | --- | --- | --- | --- |
        |  | `SAMPLE_RATE` | `MAX_ACTIVE_SECONDS`* | `MIN_FLUSH_SECONDS`* |  |
        | 모델 연산 | `MODEL_SOURCE` | `LOCAL_MODEL_DIR` | `HUB_MODEL_NAME` | `HF_CACHE_DIR` |
        |  | `LOCAL_ONLY` | `PREFER_FLOAT16`* | `compute_type`* | `LANG` |
        | 디코딩 | `beam_size`* | `patience`* | `temperature`* | `vad_filter` |
        |  | `word_timestamps`* | `task` | `initial_prompt` |  |
        | 문맥유지 | `condition_on_previous_text`* | `CONTEXT_CHARS`* | `INITIAL_GUIDE`* |  |
        | VAD | `webrtcvad.Vad(level)`* | `silence_armed` | `last_speech_ts` |  |
        | 큐, 스레드 | `audio_q.maxsize` | `decode_q.maxsize`* | `ThreadPoolExecutor(max_workers)` |  |
        | 디버깅 | `[DEBUG] push(silence) {t:.2f}s` | `[DEBUG] rtt {ms} ms` | `[FINAL] {text}` | `print_progress` |
        |  | `logging_level` |  |  |  |
        | 내부 계산 | `FRAME_SAMPLES` | `OVERLAP_SAMPLES` | `MIN_SAMPLES` | `MAX_ACTIVE_SAMPLES` |
    - 내가 수정한 파라미터
        - 오디오 세그먼트
            
            
            | `CHUNK_SECOND` | 오디오를 한번에 모델이 넣는 길이 | 0.5 ~ 30.0 | 초 | 1.8 |
            | --- | --- | --- | --- | --- |
            | `OVERLAP_SECONDS` | 인접 청크 간 겹침 시간 | 0 ~ 5.0 | 초 | 0.40 |
            | `SILENCE_TIMEOUT` | 말이 멈춘 후 ‘침묵’으로 판단하기 까지의 시간 | 0.1 ~ 1.0 | 초 | 0.30 |
            | `MAX_ACTIVE_SECONDS` | 한번에 처리 가능한 최대 발화 길이 | 1.0 ~ 60.0 | 초 | 8.0 |
            | `MIN_FLUSH_SECONDS` | 짧게 말해도 바로 전송할 최소 길이 | 0.3 ~ 2.0 | 초 | 0.60 |
        - 모델 연산
            
            
            | `PREFER_FLOAT16` | GPU에서 half precision연산 사용 | `True` / `False` | `True` |
            | --- | --- | --- | --- |
            | `compute_type` | 실제 내부 모델 연산 타입 | `int8`, `int8_float16`, `float16`, `float32` | `float16` |
        - 디코딩
            
            
            | `BEAM_SIZE` | 빔 서치 폭. 1은 가장 빠르지만 오탈자↑, 2는 균형, 5는 정확도↑ 대신 지연↑ | 1 ~ 10 | int | 2 |
            | --- | --- | --- | --- | --- |
            | `PATIENCE` | 빔 서치의 탐색 반복 제어값 | ≥ 1.0 (보통 1.0~2.0) | float | 1.0 |
            | `temperature` | 샘플링 랜덤성 제어 | 0.0 ~ 1.0 | float | 0.0 |
            | `word_timestamps` | 단어별 시간 정보를 계산할지 여부 | `True` / `False` |  | `False` |
        - 문맥 처리
            
            
            | `condition_on_previous_text` | 이전 문장의 텍스트를 다음 추론에 전달 | `True` / `False` |  | `True` |
            | --- | --- | --- | --- | --- |
            | `CONTEXT_CHARS` | 문맥 전달 시 유지할 이전 텍스트 길이 | 0 ~ 500 | 정수(글자 수) | 140 |
            | `INITIAL_GUIDE` | 첫 입력 시 모델에게 지침 | 문자열 / `None` |  | `한국어로 자연스럽게 띄어쓰고 구두점을 사용해 주세요` |
        - VAD
            
            
            | `webrtcvad.Vad()` | 음성·무음 감지 민감도 | 0, 1, 2, 3 | 1 |
            | --- | --- | --- | --- |
        - 큐, 스레드
            
            
            | `decode_q.maxsize` | 디코더 큐의 최대 크기 | 1 ~ 64 | int | 12 |
            | --- | --- | --- | --- | --- |
            | `decode_worker()` | 예외/NaN/비연속 배열 방어 로직 추가 | 사용자 정의 로직 |  |  |
        - 디버그
            
            
            | `[DEBUG] push(...)`, `[FINAL] ...` | 각 청크의 길이, 처리 완료까지의 지연(RTT), 예외 원인 실시간 확인 |
            | --- | --- |
    - 주요 파라미터
        
        
        | `FRAME_MS` | VAD(webrtcvad) 입력 프레임 길이. 오디오를 이 정도 길이로 자른 후 VAD에 전달 | ms |
        | --- | --- | --- |
        |  |  |  |

---

---

- How it works
    
    음성 입력 → `FRAME_MS` 단위로 조각냄 → `CHUNK_SECONDS` 만큼 쌓이면 청크 완성  → `OVERLAP_SECONDS` + 새로운 음성으로 청크 생성 → 디코딩 →`SILENCE_TIMEOUT` 만큼의 침묵이 지속되면 청크 생성 중지 →  whisper에게 전송 후 디코딩
    
    만약 입력의 길이가 `MAX_ACTIVE_SECONDS` 를 넘는다면 즉시 디코딩
    

---

---

- Test
    
    ![image.png](image%202.png)
    

---

---

댓글
