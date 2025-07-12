import cv2
import os

# Klasördeki tüm dosyaları al
files = os.listdir()
img_path_list = [f for f in files if f.lower().endswith(".jpg")]
print("Görüntüler:", img_path_list)

# Kedi yüzü sınıflandırıcısı
cascade_path = cv2.data.haarcascades + "haarcascade_frontalcatface.xml"
detector = cv2.CascadeClassifier(cascade_path)

# Her bir görsel için işlem yap
for img_name in img_path_list:
    print(f"İşleniyor: {img_name}")
    image = cv2.imread(img_name)
    if image is None:
        print(f"{img_name} okunamadı.")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit et
    rects = detector.detectMultiScale(
        gray,
        scaleFactor=1.045,
        minNeighbors=2,
        minSize=(50, 50)  # çok küçükleri filtrele
    )
        
    for i, (x, y, w, h) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(image, f"Kedi {i+1}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.imshow(img_name, image)
    print(f"{len(rects)} yüz bulundu.")

    # 'q' ile sıradaki görsele geç, 'esc' ile tümden çık
    key = cv2.waitKey(0)
    if key == 27:  # ESC
        break
    elif key == ord('q'):
        cv2.destroyWindow(img_name)

cv2.destroyAllWindows()
