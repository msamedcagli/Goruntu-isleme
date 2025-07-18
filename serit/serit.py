import cv2
import numpy as np

def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    
    match_mask_color = 255
    
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def drawLines(image, lines):
    
    image = np.copy(image)
    blank_image = np.zeros((image.shape[0], image.shape[1],3),dtype = np.uint8)
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(blank_image, (x1,y1),(x2,y2),(0,255,0),thickness = 10)
        
    image = cv2.addWeighted(image, 0.8, blank_image, 1, 0.0)
    return image

def process(image):
    height, width = img.shape[0], img.shape[1]
    
    region_of_interest_vertices = [(0, height), (width/2, height/2), (width, height)]
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray_image, 250, 120)
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))
    lines = cv2.HoughLinesP(cropped_image, rho = 2, theta = np.pi/180, threshold = 180, lines = np.array([]),minLineLength = 130, maxLineGap = 4)
    # print(lines)
    imageWithLine = drawLines(image, lines)
    return imageWithLine

cap = cv2.VideoCapture("video1.mp4") 
 
while True:
    success, img = cap.read()
    if not success:
        break
    img = process(img)
    cv2.imshow("img", img)
    
    # q tuşu ile çıkış
    if cv2.waitKey(20) & 0xFF == ord('q'):
        print("Kullanıcı çıkış yaptı.")
        break
     
cap.release()
cv2.destroyAllWindows()
