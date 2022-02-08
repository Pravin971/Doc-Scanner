import cv2
import numpy as np
#import stackImg

imgWidth = 640
imgHeight = 480

frameWidth = 1280
frameHeight = 720

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)  # Width 
cap.set(4, frameHeight)  # Height
cap.set(10, 30)  # Brightness

def preProcess(img):
    kernel = np.ones((5,5))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, 120, 180)
    imgDila = cv2.dilate(imgCanny, kernel, iterations = 2)
    imgThres = cv2.erode(imgDila, kernel, iterations=1)
    
    return imgThres
    
def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            # cv2.drawContours(imgContour, cnt, -1, (255,0,0), 3)
            peri = cv2.arcLength(cnt, True)   # Helps to approximate the corners of the shapes
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)  # Returns the corner points
            if area> maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    
    cv2.drawContours(imgContour, biggest, -1, (255,0,0), 20)
    return biggest


def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myNewPoints = np.zeros((4,1,2), np.int32)
    addPoints = np.sum(myPoints,1)
    
    myNewPoints[0] = myPoints[np.argmin(addPoints)]
    myNewPoints[2] = myPoints[np.argmax(addPoints)]
    
    diffPoints = np.diff(myPoints,axis=1)
    myNewPoints[1] = myPoints[np.argmin(diffPoints)]
    myNewPoints[3] = myPoints[np.argmax(diffPoints)]
    
    return myNewPoints


def getWarp(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0],[imgWidth, 0],[imgWidth, imgHeight],[0, imgHeight]])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (imgWidth,imgHeight))
    
    imgCrop = imgOutput[20:imgOutput.shape[0]-20, 20:imgOutput.shape[1]-20]
    imgCrop = cv2.resize(imgCrop, (imgWidth,imgHeight))
    return imgCrop


while True:
    suc, frame = cap.read()
    frame = cv2.resize(frame, (imgWidth, imgHeight))
    imgContour = frame.copy()
    imgThres = preProcess(frame)
    biggest = getContours(imgThres)
    imgWarp = getWarp(frame, biggest)
    
    
    
    cv2.imshow("Video", imgWarp)
    cv2.imshow("Video1", imgContour)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()