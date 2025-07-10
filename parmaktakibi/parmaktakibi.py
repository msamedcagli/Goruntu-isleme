import cv2
import time
import mediapipe as mp

# Kamera başlatma
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Kamera açılamadı.")

# Mediapipe ayarları
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# FPS için zaman değişkenleri
p_time = 0

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Kamera görüntüsü alınamadı.")
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    if id == 4:  # Baş parmak ucu
                        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        # FPS hesaplama
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        cv2.putText(img, f"FPS: {int(fps)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        cv2.imshow("Hand Tracking", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Kullanıcı tarafından durduruldu.")

finally:
    cap.release()
    cv2.destroyAllWindows()