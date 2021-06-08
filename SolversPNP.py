import cv2
import numpy as np
#import ccallib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import os

import sys
import dlib

#import Face Recognition
import face_recognition

from scipy.spatial.transform import Rotation as Rot
import Variables as V
import SimpleReturners as SRs

NumOfFP = V.NumOfFPv

#решает задачу SolvePnP для камер относительно мировой системы координат, на вход принимает изображение, его имя, матрицу камеры, дисторсию, 2d и 3d точки модели, соответствующие друг другу, возвращает позицию камеры в мировой системе координат, матрицу вращения, вектор перевода, и изображение
def WorldCamSolvePNP(img, noi, camera_matrix, dist_coeffs, two_d_model_points, three_d_model_points):
    success, rvec, tvec = cv2.solvePnP(three_d_model_points, two_d_model_points, camera_matrix, dist_coeffs)
    print ("WR Vector:\n {0}".format(rvec))
    print ("WT Vector:\n {0}".format(tvec))

    rotM = cv2.Rodrigues(rvec)[0]
    print ("WR Mat:\n {0}".format(rotM))

    cameraPosition = -(np.matrix(rotM).T) * np.matrix(tvec)

    #checking for correctness
    imagePoints, jac = cv2.projectPoints(three_d_model_points, rvec, tvec, camera_matrix, dist_coeffs)

    for i in range(4):
        cv2.circle(img,(int(imagePoints[i][0][0]),int(imagePoints[i][0][1])),2,(255,0,0),-1)

    #cv2.imshow("img", img)
    #cv2.imwrite(noi, img)
    ##############################################cv2.waitKey(0)

    return np.array(cameraPosition, dtype=np.float64), rotM, tvec, img

#возвращает 2d координаты лендмарков на изображении, на вход принимает изображение и его имя
def FacePointReturner(img, noi):
    #########faces = face_recognition.face_locations(img, model="cnn")
    # Конвертирование изображения в черно-белое
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц и построение прямоугольного контура
    faces = V.detector(gimg)

    A = list(range(6))

    if not faces:
        print("faces empty")
        return False, A, img
    else:
        #print ("Faces:\n {0}".format(faces))
        for face in faces:
        #Extracting the co cordinates to convert them into dlib rectangle object
            #x = int(face[3])
            #y = int(face[0])
            #w = int(abs(face[1]-x))
            #h = int(abs(face[2]-y))
            #u=int(face[1])
            #v=int(face[2])

            #newrect = dlib.rectangle(x,y,u,v)
            #cv2.rectangle(img, (x, y), (x+w, y+h),(0, 255, 0), 2)
            #shape = V.predictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), newrect)
            shape = V.predictor(gimg, face)

            #draw(img, shape)

            refImgPts = SRs.ref2dImagePoints(shape)
            print ("refImgPts :\n {0}".format(refImgPts))
            #for i in range(NumOfFP):
            #cv2.circle(img,(int(fimagePoints[i][0][0]),int(fimagePoints[i][0][1])),2,(0,0,255),-1)
                #cv2.putText(img, str(i), (int(refImgPts[i][0]),int(refImgPts[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

            #cv2.imshow("img", img)
            #cv2.imwrite(noi, img)
            ############################################cv2.waitKey(0)
        return True, refImgPts, img

#решает задачу SolvePnP для камер относительно системы координат лица, на вход принимает изображение, его имя, матрицу камеры, дисторсию, 2d и 3d точки модели, соответствующие друг другу, возвращает позицию камеры в системе координат лица, матрицу вращения, вектор перевода, и изображение
def FaceCamSolvePNP(img, noi, camera_matrix, dist_coeffs, two_d_model_points, three_d_model_points):
    success, rvec, tvec = cv2.solvePnP(three_d_model_points, two_d_model_points, camera_matrix, dist_coeffs)
    print ("FR Vector:\n {0}".format(rvec))
    print ("FT Vector:\n {0}".format(tvec))
    #print ("RM:\n {0}".format(rotM))

    rotM = cv2.Rodrigues(rvec)[0]
    print ("FR Mat:\n {0}".format(rotM))

    cameraPosition = -(np.matrix(rotM).T) * np.matrix(tvec)
    #print ("CP:\n {0}".format(cameraPosition))

    #checking for correctness
    imagePoints, jac = cv2.projectPoints(three_d_model_points, rvec, tvec, camera_matrix, dist_coeffs)

    for i in range(NumOfFP):
        cv2.circle(img,(int(imagePoints[i][0][0]),int(imagePoints[i][0][1])),2,(0,0,255),-1)
        #cv2.putText(img, str(i), (int(imagePoints[i][0][0]),int(imagePoints[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    #cv2.imshow("img", img)
    #cv2.imwrite(noi, img)
    ##################################################cv2.waitKey(0)

    return np.array(cameraPosition, dtype=np.float64), rotM, tvec, img