import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

tip_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    total_fingers = 0

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            lm_list = []
            h, w, _ = img.shape

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            if lm_list:
                fingers = []

                hand_label = results.multi_handedness[hand_index].classification[0].label
                # Sağ el ise baş parmak sola bakar (x küçülür), sol elde sağa bakar (x büyür)
                if hand_label == "Right":
                    if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0] - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:  # "Left"
                    if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Diğer 4 parmak (y ekseni kontrolü)
                for i in range(1, 5):
                    if lm_list[tip_ids[i]][2] < lm_list[tip_ids[i] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                total_fingers += fingers.count(1)

    # Görsel çıktı
    cv2.putText(img, str(total_fingers), (30, 125),
                cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 8)

    cv2.imshow("img", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
