import cv2
import time
import mediapipe as mp

# === Video mu, canlı kamera mı? ===
use_camera = False  # True: Canlı kamera, False: Video dosyası

if use_camera:
    cap = cv2.VideoCapture(0)  # Canlı kamera
else:
    cap = cv2.VideoCapture("video1.mp4")  # Video dosyası

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

pTime = 0

while True:
    success, img = cap.read()
    if not success:
        print("Görüntü alınamadı ya da video bitti.")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)

            # Landmark koordinatlarını görmek istersen burayı açabilirsin
            # h, w, _ = img.shape
            # for id, lm in enumerate(faceLms.landmark):
            #     cx, cy = int(lm.x * w), int(lm.y * h)
            #     print(f"ID: {id}, X: {cx}, Y: {cy}")

    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (10, 65),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Face Mesh", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Kullanıcı çıkış yaptı.")
        break

cap.release()
cv2.destroyAllWindows()
