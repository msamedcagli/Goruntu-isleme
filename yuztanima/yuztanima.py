import cv2
import mediapipe as mp

cap = cv2.VideoCapture("video4.mp4")

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.5)  # Daha sağlam algı

mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    if not success:
        print("Video bitti veya görüntü alınamadı.")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        h, w, _ = img.shape
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            
            # Sınırları görsel olarak biraz temizle
            x, y, width, height = bbox
            x, y = max(0, x), max(0, y)
            x2, y2 = x + width, y + height
            
            cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 255), 2)

    cv2.imshow("Face Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Kullanıcı çıkış yaptı.")
        break

cap.release()
cv2.destroyAllWindows()
