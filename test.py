import cv2
import numpy as np
import time
import imutils
import sys
import easyocr
import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#pip uninstall opencv-python-headless -> [Other Error]
#pip uninstall opencv-python; pip install opencv-python 

cap = cv2.VideoCapture('videoplayback.mp4')
fps= int(cap.get(cv2.CAP_PROP_FPS))
fpsJump = fps
print("FPS: ", fps)
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
while(cap.isOpened()):
  cap.set(cv2.CAP_PROP_POS_FRAMES, fpsJump)
  #start_time = time.time()
  ret, frame = cap.read()
  
  if ret == True:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
    edged = cv2.Canny(bfilter, 30, 200) #Edge detection
    cv2.imshow('edges',cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    flag = 0
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            flag = 1
            location = approx
            break
    if flag==1:
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0,255, -1)
        new_image = cv2.bitwise_and(frame, frame, mask=mask)
        #cv2.imshow("Croped",cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        (x,y) = np.where(mask==255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]
        #cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Croped Plate",cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        if result:
          numberPlate = result[0][-2]
          print(numberPlate)
        else:
          print("No text found!")
        #print(result)
    fpsJump = fpsJump + 5
    #time.sleep(1)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  else: 
    break

# When everything done, release the video capture object
