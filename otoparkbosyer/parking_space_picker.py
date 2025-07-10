import cv2
import pickle
import os

# Park yeri boyutları
width, height = 27, 15

# Kayıtlı pozisyonları yükle
carpark_file = "CarParkPos"

if os.path.exists(carpark_file):
    with open(carpark_file, "rb") as f:
        posList = pickle.load(f)
else:
    posList = []

# Mouse olayı işleyici
def mouseClick(events, x, y, flags, params):
    global posList
    changed = False

    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
        changed = True

    elif events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)
                changed = True
                break

    if changed:
        with open(carpark_file, "wb") as f:
            pickle.dump(posList, f)
        print(f"Pozisyonlar güncellendi: {len(posList)} alan işaretli.")

# Görseli yükle
img_path = "first_frame.png"
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Görsel bulunamadı: {img_path}")

while True:
    img = cv2.imread(img_path)

    for pos in posList:
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (255, 0, 0), 2)

    cv2.imshow("Park Yeri Seçimi", img)
    cv2.setMouseCallback("Park Yeri Seçimi", mouseClick)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Etiketleme sona erdi.")
        break

cv2.destroyAllWindows()
