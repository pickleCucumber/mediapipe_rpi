import cv2
import time
from picamera2 import Picamera2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Инициализация MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

DETECTION_RESULT = None
FPS = 0
START_TIME = time.time()

def save_result(result, output_image, timestamp_ms):
    global DETECTION_RESULT, FPS, START_TIME
    DETECTION_RESULT = result
    FPS = 1 / (time.time() - START_TIME)
    START_TIME = time.time()

def main():
    # Инициализация камеры
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    # Настройки детектора
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=save_result)

    with vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            # Получаем кадр
            frame = picam2.capture_array()
            
            # Конвертируем в RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Детекция
            detector.detect_async(mp_image, int(time.time_ns() / 1_000_000))
            
            # Отрисовка FPS
            cv2.putText(frame, f"FPS: {FPS:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Отрисовка результатов
            if DETECTION_RESULT and DETECTION_RESULT.hand_landmarks:
                for hand_landmarks in DETECTION_RESULT.hand_landmarks:
                    # Создаем NormalizedLandmarkList
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                        for lm in hand_landmarks
                    ])
                    
                    # Отрисовываем 
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks_proto,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            
            cv2.imshow('Hand Tracking', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
