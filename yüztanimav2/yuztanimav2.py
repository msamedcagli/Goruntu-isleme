import cv2
import matplotlib.pyplot as plt

# Haar cascade sınıflandırıcıyı yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Einstein resmi
einstein = cv2.imread("einstein.jpg", 0)
plt.figure(), plt.imshow(einstein, cmap="gray"), plt.axis("off")

faces = face_cascade.detectMultiScale(einstein, scaleFactor=1.1, minNeighbors=5)

einstein_detected = einstein.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(einstein_detected, (x, y), (x+w, y+h), (255, 255, 255), 10)

plt.figure(), plt.imshow(einstein_detected, cmap="gray"), plt.axis("off")

# Barcelona resmi
barce = cv2.imread("barcelona.jpg", 0)
plt.figure(), plt.imshow(barce, cmap="gray"), plt.axis("off")

faces = face_cascade.detectMultiScale(barce, scaleFactor=1.1, minNeighbors=7)

barce_detected = barce.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(barce_detected, (x, y), (x+w, y+h), (255, 255, 255), 10)

plt.figure(), plt.imshow(barce_detected, cmap="gray"), plt.axis("off")

# Gerçek zamanlı video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera açılamadı!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Kare alınamadı.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

    cv2.imshow("Face Detection (Q ile çık)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Çıkış yapılıyor...")
        break

cap.release()
cv2.destroyAllWindows()
