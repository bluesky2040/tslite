# ============================================================
# EfficientDet-Lite0 + Raspberry Pi 5 + PiCamera2
#    실시간 객체 탐지 (Object Detection) 실습용
#
# [실행 전 준비]
#   pip install picamera2 opencv-python tensorflow numpy
#   모델 파일: lite-model_efficientdet_lite0_detection_default_1.tflite
#   레이블 파일: COCO2017_classes.txt
#
# [실행 방법]
#   python test_tflite6.py
#
# [종료 방법]
#   화면에서 'q' 키를 누르거나 Ctrl+C
# ============================================================

from picamera2 import Picamera2
import cv2
import numpy as np
import tensorflow as tf
import time
import sys
import os


# ──────────────────────────────────────────
# 1. 설정값 (한 곳에서 수정하기 쉽게 분리)
# ──────────────────────────────────────────
IMG_SIZE        = 320          # 모델 입력 해상도 (320x320 고정)
SCORE_THRESHOLD = 0.5          # 탐지 신뢰도 기준 (0.0 ~ 1.0)
                               #   낮추면 → 더 많이 탐지 (오탐 증가)
                               #   높이면 → 더 엄격하게 탐지 (미탐 증가)
MODEL_PATH      = './lite-model_efficientdet_lite0_detection_default_1.tflite'
LABEL_PATH      = './COCO2017_classes.txt'
BOX_COLOR       = (0, 255, 0)  # 박스 색상 (BGR 순서: 초록)
TEXT_COLOR      = (0, 255, 0)  # 레이블 텍스트 색상
FONT            = cv2.FONT_HERSHEY_SIMPLEX


# ──────────────────────────────────────────
# 2. 레이블 로드 함수
# ──────────────────────────────────────────
def load_labels(label_path: str) -> list:
    """
    COCO 클래스 레이블 파일을 읽어 리스트로 반환합니다.
    예: ['person', 'bicycle', 'car', ...]

    Args:
        label_path: 레이블 파일 경로
    Returns:
        labels: 클래스 이름 리스트
    """
    if not os.path.exists(label_path):
        print(f"[ERROR] Label file not found: {label_path}")
        sys.exit(1)

    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            for word in line.split(','):
                labels.append(word.strip().lower())

    print(f"[INFO] Loaded {len(labels)} labels successfully")
    return labels


# ──────────────────────────────────────────
# 3. TFLite 모델 로드 함수
# ──────────────────────────────────────────
def load_model(model_path: str):
    """
    TensorFlow Lite 모델을 로드하고 인터프리터를 반환합니다.
    입력/출력 텐서 정보도 함께 반환합니다.

    Args:
        model_path: TFLite 모델 파일 경로
    Returns:
        interpreter   : TFLite 인터프리터 객체
        input_details : 입력 텐서 정보
        output_details: 출력 텐서 정보
        input_dtype   : 입력 데이터 타입 (uint8 또는 float32)
    """
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 모델의 입력 타입 확인 (uint8 / float32)
    input_dtype = input_details[0]['dtype']
    print(f"[INFO] Model input dtype : {input_dtype.__name__}")
    print(f"[INFO] Model input shape : {input_details[0]['shape']}")

    return interpreter, input_details, output_details, input_dtype


# ──────────────────────────────────────────
# 4. 카메라 초기화 함수
# ──────────────────────────────────────────
def init_camera(img_size: int) -> Picamera2:
    """
    PiCamera2를 초기화하고 시작합니다.

    Args:
        img_size: 캡처 해상도 (정사각형, 예: 320)
    Returns:
        picam2: 초기화된 카메라 객체
    """
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (img_size, img_size), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)  # 카메라 워밍업 대기 (1초)
    print("[INFO] Camera started successfully")
    return picam2


# ──────────────────────────────────────────
# 5. 전처리 함수 (입력 타입에 따라 자동 변환)
# ──────────────────────────────────────────
def preprocess(frame: np.ndarray, input_dtype) -> np.ndarray:
    """
    카메라 프레임을 모델 입력 형식으로 변환합니다.

    변환 규칙:
        - uint8  모델: 0~255 그대로 사용
        - float32 모델: 0.0~1.0 으로 정규화

    Args:
        frame      : 카메라에서 받은 원본 이미지 (H, W, 3)
        input_dtype: 모델이 기대하는 입력 데이터 타입
    Returns:
        img: 모델 입력용 배열 (1, H, W, 3)
    """
    img = np.array(frame)           # (320, 320, 3)
    img = img[np.newaxis, :, :, :]  # (1, 320, 320, 3) 배치 차원 추가

    if input_dtype == np.float32:
        img = img.astype(np.float32) / 255.0  # 정규화: 0~255 → 0.0~1.0
    else:
        img = img.astype(np.uint8)

    return img


# ──────────────────────────────────────────
# 6. 탐지 결과 그리기 함수
# ──────────────────────────────────────────
def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    labels: list,
    threshold: float,
    img_size: int
) -> tuple:
    """
    탐지된 객체를 이미지에 박스 + 레이블로 그립니다.

    Args:
        image    : 원본 이미지
        boxes    : 탐지 박스 좌표 배열 [(ymin,xmin,ymax,xmax), ...]
        classes  : 탐지된 클래스 ID 배열
        scores   : 탐지 신뢰도 배열
        labels   : 클래스 이름 리스트
        threshold: 신뢰도 최소 기준
        img_size : 이미지 크기 (픽셀)
    Returns:
        out_img        : 박스/레이블이 그려진 이미지
        detected_count : 실제 탐지된 객체 수
    """
    out_img = image.copy()
    detected_count = 0

    # scores 기준 내림차순 정렬 (신뢰도 높은 것부터 처리)
    sorted_indices = np.argsort(scores)[::-1]

    for idx in sorted_indices:
        score = scores[idx]

        # 신뢰도 기준 미달이면 중단
        if score < threshold:
            break

        # 박스 좌표 계산: 비율(0~1) → 픽셀 좌표
        ymin, xmin, ymax, xmax = boxes[idx]
        xmin = int(max(1,        xmin * img_size))
        xmax = int(min(img_size, xmax * img_size))
        ymin = int(max(1,        ymin * img_size))
        ymax = int(min(img_size, ymax * img_size))

        # 레이블 텍스트 준비 (영어 클래스명 + 신뢰도)
        class_id   = int(classes[idx])
        label_name = labels[class_id] if class_id < len(labels) else f"id:{class_id}"
        label_text = f"{label_name}: {score:.2f}"

        # 박스 그리기
        cv2.rectangle(out_img, (xmin, ymin), (xmax, ymax), BOX_COLOR, 2)

        # 텍스트 배경 (가독성을 위한 검정 배경)
        (tw, th), _ = cv2.getTextSize(label_text, FONT, 0.5, 1)
        cv2.rectangle(
            out_img,
            (xmin, ymin - th - 6),
            (xmin + tw, ymin),
            (0, 0, 0), -1  # -1: 채우기
        )

        # 레이블 텍스트 출력 (화면 표시는 영어)
        cv2.putText(
            out_img, label_text,
            (xmin, ymin - 4),
            FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA
        )

        detected_count += 1

    return out_img, detected_count


# ──────────────────────────────────────────
# 7. FPS 측정 클래스
# ──────────────────────────────────────────
class FPSCounter:
    """
    초당 프레임 수(FPS)를 측정하고 화면에 표시하는 클래스.
    1초마다 FPS 값을 갱신합니다.
    """
    def __init__(self):
        self._start = time.time()
        self._count = 0
        self.fps    = 0.0

    def update(self):
        """프레임이 처리될 때마다 호출"""
        self._count += 1
        elapsed = time.time() - self._start
        if elapsed >= 1.0:              # 1초마다 FPS 갱신
            self.fps    = self._count / elapsed
            self._count = 0
            self._start = time.time()

    def draw(self, image: np.ndarray) -> np.ndarray:
        """FPS를 이미지 좌측 상단에 표시 (화면 표시는 영어)"""
        text = f"FPS: {self.fps:.1f}"
        cv2.putText(
            image, text, (10, 25),
            FONT, 0.7, (255, 255, 0), 2, cv2.LINE_AA
        )
        return image


# ──────────────────────────────────────────
# 8. 메인 실행 함수
# ──────────────────────────────────────────
def main():
    print("=" * 50)
    print("  EfficientDet-Lite0 Real-time Object Detection")
    print(f"  Confidence Threshold : {SCORE_THRESHOLD}")
    print("  Press 'q' to quit")
    print("=" * 50)

    # ── 초기화 ──
    labels = load_labels(LABEL_PATH)
    interpreter, input_details, output_details, input_dtype = load_model(MODEL_PATH)
    picam2      = init_camera(IMG_SIZE)
    fps_counter = FPSCounter()

    try:
        while True:
            # Step 1: 프레임 캡처
            frame = picam2.capture_array()

            # Step 2: 전처리 (모델 입력 타입에 맞게 자동 변환)
            input_tensor = preprocess(frame, input_dtype)

            # Step 3: 모델 추론 (inference)
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()

            # Step 4: 결과 추출
            boxes   = interpreter.get_tensor(output_details[0]['index']).squeeze()
            classes = interpreter.get_tensor(output_details[1]['index']).squeeze()
            scores  = interpreter.get_tensor(output_details[2]['index']).squeeze()
            # output_details[3]: count (유효 탐지 수) → draw_detections에서 직접 카운트

            # Step 5: 탐지 결과 이미지에 그리기
            out_img, det_count = draw_detections(
                np.array(input_tensor[0]),  # 표시용 이미지 (배치 차원 제거)
                boxes, classes, scores,
                labels, SCORE_THRESHOLD, IMG_SIZE
            )

            # Step 6: FPS 측정 및 화면 표시
            fps_counter.update()
            fps_counter.draw(out_img)

            # Step 7: 탐지 수 화면 표시 (영어)
            cv2.putText(
                out_img, f"Detected: {det_count}",
                (10, 50), FONT, 0.7, (0, 200, 255), 2, cv2.LINE_AA
            )

            # Step 8: 결과 화면 출력
            cv2.imshow("EfficientDet-Lite0 Detection", out_img)

            # Step 9: 콘솔 로그 (탐지된 경우만 출력, 영어)
            if det_count > 0:
                print(f"[DETECT] {det_count} object(s) found | FPS: {fps_counter.fps:.1f}")

            # Step 10: 'q' 키 입력 시 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Quit requested by user.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by Ctrl+C.")

    finally:
        # 항상 자원 해제 (카메라 & 윈도우)
        picam2.stop()
        cv2.destroyAllWindows()
        print("[INFO] Camera and windows closed.")


# ──────────────────────────────────────────
# 9. 진입점
# ──────────────────────────────────────────
if __name__ == '__main__':
    main()
