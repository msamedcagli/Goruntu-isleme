import cv2
import matplotlib.pyplot as plt

# Görüntüyü oku ve renkli/gri formatlara çevir
img_color = cv2.cvtColor(cv2.imread("isokartal.jpeg"), cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)

# Sabit threshold
_, thresh = cv2.threshold(img_gray, 60, 255, cv2.THRESH_BINARY)

# Adaptif threshold
adaptive_thresh = cv2.adaptiveThreshold(
    img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8
)

# Görsel başlıkları ve içerikleri
titles = ["Orijinal (Renkli)", "Gri Ton", "Sabit Threshold", "Adaptive Threshold"]
images = [img_color, img_gray, thresh, adaptive_thresh]

# Görselleri yanyana göster
plt.figure(figsize=(16, 4))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(images[i], cmap="gray" if i > 0 else None)
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
