import cv2

# içe aktarma
img = cv2.imread("vettel.jpg", 0)

# görselleştir
cv2.imshow("deneme", img)


k = cv2.waitKey(0) &0xFF

if k == 27: #esc tuşu
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite("vettelgray.png", img)
    cv2.destroyAllWindows()