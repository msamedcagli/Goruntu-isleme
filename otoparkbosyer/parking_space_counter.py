import cv2
import pickle
import numpy as np

# === Ayarlar ===
width, height = 27, 15
video_path = "video.mp4"
carpark_file = "CarParkPos"

# === Yüzey tanıma fonksiyonu ===
def checkParkSpace(processed_img, original_img):
    spaceCounter = 0

    for pos in posList:
        x, y = pos
        img_crop = processed_img[y:y + height, x:x + width]
        count = cv2.countNonZero(img_crop)

        if count < 150:
            color = (0, 255, 0)
            spaceCounter += 1
        else:
            color = (0, 0, 255)

        cv2.rectangle(original_img, pos, (x + width, y + height), color, 2)
        cv2.putText(original_img, str(count), (x, y + height - 2), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

    cv2.putText(original_img, f"Bos Park:{spaceCounter}", (15, 24),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)

# === Yüklemeler ===
cap = cv2.VideoCapture(video_path)

with open(carpark_file, "rb") as f:
    posList = pickle.load(f)

# === Ana döngü ===
while True:
    success, img = cap.read()
    if not success:
        print("Video bitti veya oynatılamıyor.")
        break

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThresh = cv2.adaptiveThreshold(imgBlur, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThresh, 5)
    imgDilate = cv2.dilate(imgMedian, np.ones((3, 3), np.uint8), iterations=1)

    checkParkSpace(imgDilate, img)

    cv2.imshow("Parking Detection", img)

    # 'q' tuşuyla çıkış
    if cv2.waitKey(200) & 0xFF == ord('q'):
        print("Kullanıcı çıkış yaptı.")
        break

cap.release()
cv2.destroyAllWindows()
