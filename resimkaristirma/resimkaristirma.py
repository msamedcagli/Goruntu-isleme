import cv2
import matplotlib.pyplot as plt

# Görselleri oku ve boyutlandır
img1 = cv2.imread("isokartal.jpeg")
img2 = cv2.imread("stad.jpeg")

img1 = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), (800, 800))
img2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), (800, 800))

# Görselleri karıştır
blended = cv2.addWeighted(img1, 0.8, img2, 0.2, 0)

# Görseli göster ve kaydet
plt.imshow(blended)
plt.axis('off')
plt.savefig("karistirilanresim.jpg", bbox_inches='tight', pad_inches=0)
plt.show()
