import cv2

#%% Orjinal
img = cv2.imread("vettel.jpg")
cv2.imshow("Orjinal", img)
print("Resim boyutu: ", img.shape)

#%% Boyutlandırma
imgResized = cv2.resize(img,(1024,1024))
cv2.imshow("Boyutlanan Resim", imgResized)
print("Yeniden boyutlandırılan resim boyutu: ",imgResized.shape)

#%% Kırpma
imgCropped = img[0:400,0:400] 
cv2.imshow("Kirpilan Resim", imgCropped)
print("Kırpılan resim boyutu: ", imgCropped.shape)