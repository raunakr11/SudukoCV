#Solving Suduko using OpenCV
#---------------------------

print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    #for getting rid of all warning messages
from utils import *

#libraries
import cv2 as cv
import numpy as np
import sudukoSolver
from utils import *
import pygame

#image path and resolution
pathImage = "Resources/2.png"
heightImg = 450
widthImg = 450
model = intializePredectionModel()

# model = intializePredectionModel()  # LOAD THE CNN MODEL

#### preparing the image
img = cv.imread(pathImage)
img = cv.resize(img, (widthImg, heightImg))  # resizing to square
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
imgThreshold = preProcess(img)

# #### 2. FIND ALL COUNTOURS
imgContours = img.copy() # will contain all contours
imgBigContour = img.copy() # will contain biggest contour
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # (img, method of contour(because outer contours), chain approximation)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # (image, contours, contourIdx, color, thickness)

#### 3. FIND THE BIGGEST COUNTOUR AND USE IT AS SUDOKU
biggest, maxArea = biggestContour(contours) # FIND THE BIGGEST CONTOUR
# print(biggest)
if biggest.size != 0:
    biggest = reorder(biggest)
    # print(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25) # draws the biggest contour
    pts1 = np.float32(biggest) #the points returned from reorder()
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) #pre defined points
    matrix = cv.getPerspectiveTransform(pts1, pts2) # GER
    imgWarpColored = cv.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)


 #### 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    # print(len(boxes))
    # cv2.imshow("Sample",boxes[78])
    numbers = getPredection(boxes, model)
    # print(numbers)
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)  #empty spaces are 1, rest 0
    # print(posArray)         

     #### 5. FIND SOLUTION OF THE BOARD
    board = np.array_split(numbers,9)
    # print(board)  #board before solving
    try:
        sudukoSolver.solve(board)
    except:
        pass
    # print(board)  #board after solving
    flatList = []
    for sublist in board:   #to convert it into flatlist
        for item in sublist:
            flatList.append(item)
    solvedNumbers =flatList*posArray    #to show only 
    imgSolvedDigits= displayNumbers(imgSolvedDigits,solvedNumbers)

    # #### 6. OVERLAY SOLUTION
    pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    imgDetectedDigits = drawGrid(imgDetectedDigits)
    imgSolvedDigits = drawGrid(imgSolvedDigits)

    imageArray = ([img,imgThreshold,imgContours, imgBigContour],
                  [imgDetectedDigits, imgSolvedDigits,imgInvWarpColored,inv_perspective])
    stackedImage = stackImages(imageArray, 1)
    cv2.imshow('Stacked Images', stackedImage)

#stores all images to view them step by step
imageArray = ([img,imgThreshold,imgContours, imgBigContour],
                  [imgWarpColored, imgDetectedDigits, imgSolvedDigits, inv_perspective])
stackedImage = stackImages(imageArray, 1)
cv.imshow('Stacked Images', stackedImage)

cv.waitKey(0)