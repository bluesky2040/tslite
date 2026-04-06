# EfficientDet-Lite0 실시간 객체 탐지 (Raspberry Pi 5)

**파일명:** `test_tflite6.py`  
**대상 환경:** Raspberry Pi 5 + PiCamera2  
**목적:** TFLite 모델을 이용한 실시간 객체 탐지 (Object Detection) 실습

---

## TensorFlow Lite (TFLite) 란?

### 개요
TensorFlow Lite는 Google이 개발한 **엣지 디바이스(Edge Device) 전용 경량 딥러닝 프레임워크**입니다.
스마트폰, 라즈베리파이, 마이크로컨트롤러 등 연산 자원이 제한된 환경에서 AI 모델을 실행할 수 있도록 설계되었습니다.

---

### TensorFlow vs TensorFlow Lite 비교

| 항목 | TensorFlow (일반) | TensorFlow Lite |
|------|------------------|-----------------|
| 주요 용도 | 모델 학습 + 추론 | 추론(Inference) 전용 |
| 실행 환경 | PC, 서버, GPU | 모바일, 임베디드, IoT |
| 파일 형식 | SavedModel, HDF5 | `.tflite` (FlatBuffer) |
| 파일 크기 | 수백 MB ~ GB | 수 MB 이하 |
| 연산 속도 | 빠름 (GPU 활용) | 최적화된 경량 연산 |
| 메모리 사용 | 많음 | 매우 적음 |

---

### TFLite 동작 원리

```
[학습 서버 / PC]                        [엣지 디바이스 / Raspberry Pi]
─────────────────────                   ──────────────────────────────
TensorFlow 모델 학습                     .tflite 파일 로드
      ↓                                       ↓
TFLite Converter로 변환          →     Interpreter 생성
      ↓                                       ↓
.tflite 파일 생성                        allocate_tensors()
                                              ↓
                                    입력 텐서에 데이터 주입
                                    set_tensor()
                                              ↓
                                    추론 실행
                                    invoke()
                                              ↓
                                    출력 텐서에서 결과 추출
                                    get_tensor()
```

---

### .tflite 파일 구조
`.tflite` 파일은 **FlatBuffer** 포맷으로 저장된 이진 파일입니다.

| 구성 요소 | 설명 |
|-----------|------|
| 모델 그래프 | 연산(Op)들의 연결 구조 |
| 가중치(Weights) | 학습된 파라미터 값 |
| 텐서 정보 | 입출력 shape, dtype |
| 메타데이터 | 버전, 라벨 정보 등 |

---

### 양자화 (Quantization)
TFLite의 핵심 경량화 기법으로, 모델의 가중치와 연산을 낮은 정밀도로 변환합니다.

| 방식 | 데이터 타입 | 파일 크기 | 속도 | 정확도 |
|------|------------|-----------|------|--------|
| 원본 (Full Precision) | float32 (32bit) | 기준 | 기준 | 기준 |
| 동적 양자화 | int8 (8bit) | 약 1/4 | 빠름 | 약간 낮음 |
| 전체 정수 양자화 | uint8 (8bit) | 약 1/4 | 가장 빠름 | 약간 낮음 |
| 부동소수점 16bit | float16 (16bit) | 약 1/2 | 빠름 | 거의 동일 |

> 이 실습에서 사용하는 `EfficientDet-Lite0` 모델의 입력 타입이 `uint8`인 이유가 바로 **전체 정수 양자화** 적용 때문입니다.  
> 코드에서 `input_dtype`을 확인하여 `uint8` / `float32`를 자동 분기하는 것도 이 때문입니다.

---

### TFLite Interpreter (인터프리터)
코드에서 사용하는 핵심 객체로, 모델 실행의 전 과정을 담당합니다.

```python
# 1. 인터프리터 생성 및 메모리 할당
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# 2. 입출력 텐서 정보 조회
input_details  = interpreter.get_input_details()   # 입력 텐서 목록
output_details = interpreter.get_output_details()  # 출력 텐서 목록

# 3. 입력 데이터 주입
interpreter.set_tensor(input_details[0]['index'], input_tensor)

# 4. 추론 실행
interpreter.invoke()

# 5. 결과 추출
boxes = interpreter.get_tensor(output_details[0]['index'])
```

| 메서드 | 역할 |
|--------|------|
| `Interpreter(model_path)` | .tflite 파일 로드 |
| `allocate_tensors()` | 입출력 텐서 메모리 할당 (한 번만 호출) |
| `get_input_details()` | 입력 텐서의 index, shape, dtype 조회 |
| `get_output_details()` | 출력 텐서의 index, shape, dtype 조회 |
| `set_tensor(index, data)` | 입력 텐서에 데이터 주입 |
| `invoke()` | 실제 추론 실행 |
| `get_tensor(index)` | 출력 텐서에서 결과 읽기 |

---

### XNNPACK 가속
Raspberry Pi에서 실행 시 아래 메시지가 출력됩니다:
```
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
```
**XNNPACK**은 ARM CPU(Raspberry Pi 포함)에서 부동소수점 연산을 최적화하는 가속 라이브러리입니다.
자동으로 활성화되며, 추론 속도를 크게 향상시킵니다.

---

### 지원 플랫폼

| 플랫폼 | 지원 여부 |
|--------|----------|
| Android | O |
| iOS | O |
| Raspberry Pi (Linux ARM) | O |
| Windows / macOS / Linux (x86) | O |
| Arduino / Microcontroller | O (TFLite Micro) |

---

## 필요 파일

| 파일 | 설명 |
|------|------|
| `test_tflite6.py` | 메인 실행 스크립트 |
| `lite-model_efficientdet_lite0_detection_default_1.tflite` | TFLite 객체 탐지 모델 |
| `COCO2017_classes.txt` | COCO 91개 클래스 레이블 (쉼표 구분) |

---

## 환경 설정

### 패키지 설치
```bash
pip install picamera2 opencv-python tensorflow numpy
```

### 하드웨어 요구사항
- Raspberry Pi 5
- PiCamera2 (CSI 카메라 모듈)

---

## 실행 방법

```bash
python test_tflite6.py
```

### 종료 방법
- 화면에서 `q` 키
- 터미널에서 `Ctrl+C`

---

## 설정값 (코드 상단에서 수정)

```python
IMG_SIZE        = 320    # 모델 입력 해상도 (320x320 고정)
SCORE_THRESHOLD = 0.5    # 탐지 신뢰도 기준 (0.0 ~ 1.0)
MODEL_PATH      = './lite-model_efficientdet_lite0_detection_default_1.tflite'
LABEL_PATH      = './COCO2017_classes.txt'
BOX_COLOR       = (0, 255, 0)   # 박스 색상 (BGR, 초록)
TEXT_COLOR      = (0, 255, 0)   # 텍스트 색상 (BGR, 초록)
```

| 설정값 | 설명 |
|--------|------|
| `SCORE_THRESHOLD` 낮추기 | 더 많이 탐지 (오탐 증가) |
| `SCORE_THRESHOLD` 높이기 | 더 엄격하게 탐지 (미탐 증가) |

---

## 소스 구조

```
test_tflite6.py
├── 1. 설정값
├── 2. load_labels()      레이블 파일 로드
├── 3. load_model()       TFLite 모델 로드
├── 4. init_camera()      PiCamera2 초기화
├── 5. preprocess()       프레임 전처리
├── 6. draw_detections()  탐지 결과 시각화
├── 7. FPSCounter 클래스  FPS 측정
└── 8. main()             메인 루프
```

---

## 주요 함수 설명

### `load_labels(label_path)` — 레이블 로드
- `COCO2017_classes.txt`를 읽어 클래스 이름 리스트 반환
- 파일 형식: 쉼표(`,`) 구분, 1줄로 작성
- 반환 예: `['person', 'bicycle', 'car', ...]`

### `load_model(model_path)` — 모델 로드
- TFLite 인터프리터를 초기화하고 텐서 메모리 할당
- 입력 텐서 타입 자동 감지 (`uint8` 또는 `float32`)
- 반환값: `(interpreter, input_details, output_details, input_dtype)`

### `init_camera(img_size)` — 카메라 초기화
- PiCamera2를 `img_size × img_size` 해상도, `RGB888` 포맷으로 설정
- 워밍업을 위해 1초 대기 후 반환

### `preprocess(frame, input_dtype)` — 전처리
- PiCamera2 프레임 `(H, W, 3)` → 모델 입력 `(1, H, W, 3)` 배치 차원 추가
- 입력 타입에 따라 자동 변환:
  - `uint8`: 그대로 사용 (0 ~ 255)
  - `float32`: 정규화 (0.0 ~ 1.0)

### `draw_detections(image, boxes, classes, scores, labels, threshold, img_size)` — 시각화
- 신뢰도 기준(`threshold`) 이상인 탐지 결과만 표시
- 신뢰도 높은 순으로 정렬하여 처리
- 각 탐지 객체에 바운딩 박스 + 클래스명 + 신뢰도 표시
- 박스 좌표: 비율값(0~1) → 픽셀 좌표 변환 (`ymin, xmin, ymax, xmax`)

### `FPSCounter` 클래스 — FPS 측정
- `update()`: 매 프레임마다 호출, 1초마다 FPS 갱신
- `draw(image)`: 이미지 좌측 상단에 FPS 표시 (노란색)

---

## 메인 루프 처리 순서

```
while True:
  Step 1  프레임 캡처          picam2.capture_array()
  Step 2  전처리               preprocess()
  Step 3  모델 추론            interpreter.invoke()
  Step 4  결과 추출            boxes / classes / scores
  Step 5  탐지 결과 그리기     draw_detections()
  Step 6  FPS 표시             fps_counter.draw()
  Step 7  탐지 수 표시         cv2.putText()
  Step 8  화면 출력            cv2.imshow()
  Step 9  콘솔 로그            탐지된 경우만 출력
  Step 10 종료 키 확인         'q' 입력 시 break
```

---

## 모델 출력 텐서 구성

EfficientDet-Lite0 모델은 4개의 출력 텐서를 반환합니다.

| 인덱스 | 내용 | shape |
|--------|------|-------|
| `output_details[0]` | 바운딩 박스 좌표 `(ymin, xmin, ymax, xmax)` | `[1, 25, 4]` |
| `output_details[1]` | 클래스 ID | `[1, 25]` |
| `output_details[2]` | 신뢰도 (score) | `[1, 25]` |
| `output_details[3]` | 유효 탐지 수 (count) | `[1]` |

> 최대 25개 객체까지 탐지 결과를 반환하며, `SCORE_THRESHOLD` 이상인 것만 화면에 표시합니다.

---

## 화면 표시 항목

| 항목 | 위치 | 색상 |
|------|------|------|
| FPS | 좌측 상단 (10, 25) | 노란색 |
| 탐지 수 (Detected: N) | 좌측 상단 (10, 50) | 하늘색 |
| 바운딩 박스 | 객체 위치 | 초록색 |
| 클래스명 + 신뢰도 | 박스 상단 | 초록색 (검정 배경) |

---

## 탐지 가능한 클래스 (COCO 80개)

person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench,
bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe,
backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard,
sports ball, kite, baseball bat, baseball glove, skateboard, surfboard,
tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl,
banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza,
donut, cake, chair, couch, potted plant, bed, dining table, toilet,
tv, laptop, mouse, remote, keyboard, cell phone,
microwave, oven, toaster, sink, refrigerator, book, clock, vase,
scissors, teddy bear, hair drier, toothbrush

> `COCO2017_classes.txt`에는 원본 COCO ID 기준으로 빈 슬롯(`???`)을 포함한 91개 항목이 저장되어 있습니다.

---

## 자원 해제

프로그램 종료 시 `finally` 블록에서 항상 실행됩니다.

```python
picam2.stop()           # 카메라 스트림 중단
cv2.destroyAllWindows() # OpenCV 윈도우 닫기
```
