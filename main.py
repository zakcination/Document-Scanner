import shutil
import cv2
import numpy as np
import utlis
import os
from datetime import datetime

webCamFeed = True
pathImage = ["sample1.jpg", "sample2.jpeg", "sample3.jpeg"]
cap = cv2.VideoCapture(0)
cap.set(10, 160)
# cap.set(3, 1280)
# cap.set(4, 720)
heightImg = 480
widthImg = 640


utlis.initializeTrackbars()
count = 0
while True:

    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    if webCamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage[2])

    img = cv2.resize(img, (widthImg, heightImg))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    thres = utlis.valTrackbars()
    imgThreshold = cv2.Canny(imgBlur,250,1) # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

    # find countours
    imgContours = img.copy()
    imgBigContour = img.copy()
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    # find the biggest countour
    biggest, maxArea = utlis.biggestCountour(contours)
    if biggest.size != 0:
        biggest = utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
        imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # REMOVE 20 PIXELS FORM EACH SIDE
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

        # APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)
        #imgAdaptiveThre = cv2.threshold(imgWarpGray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Image Array for Display
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])

    else:
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    # LABELS FOR DISPLAY
    lables = [["Original", "Gray", "Threshold", "Contours"],
              ["Biggest Contour", "Warp Prespective", "Warp Gray", "Adaptive Threshold"]]

    stackedImage = utlis.stackImages(imageArray, 0.75, lables)
    cv2.imshow("Result", stackedImage)

    # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        path = "Scanned/scan" + str(count)
        isExist = os.path.exists(path)
        if isExist:
            shutil.rmtree(path)
        os.makedirs(path)
        dt = datetime.now()
        str_date_time = dt.strftime("%d-%m-%Y_%H:%M:%S")
        print(cv2.imwrite(path + '/image' + str(count) + "_original_" + f'{str_date_time}' + ".jpg", img))
        print(cv2.imwrite(path + '/image' + str(count) + "_contour_" + f'{str_date_time}' + ".jpg", imgBigContour))
        print(cv2.imwrite(path + '/image' + str(count) + "_warpColored_" + f'{str_date_time}' + ".jpg", imgWarpColored))
        print(cv2.imwrite(path + '/image' + str(count) + "_warpGray_" + f'{str_date_time}' + ".jpg", imgWarpGray))
        print(cv2.imwrite(path + '/image' + str(count) + "_adaptiveThreshold_" + f'{str_date_time}' + ".jpg", imgAdaptiveThre))
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(1000)
        count += 1
