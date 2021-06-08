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

#смотри ниже
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
#эта функция, как и предыдущая нужна для того, чтобы определить углы вращения по матрице вращения
def rotationMatrixToEulerAngles(R):
 
    assert(isRotationMatrix(R))
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([z, y, x])

def FaceToWorldTranslator(img, noi, camera_matrix, dist_coeffs, world_cam_rotM, world_cam_trV, face_cam_rotM, face_cam_trV):
    RotatedfaceCord = list(range(NumOfFP))

    for i in range(NumOfFP):
        FaceCord = V.face3Dmodel[i]
        #    print ("fc:\n {0}".format(faceCord))
        #nFaceCord = (np.matrix(rotation_matrix)*np.matrix(faceCord).T).T#########################################np.matrix(world_cam_rotM)*
        nFaceCord = (np.matrix(face_cam_rotM)*np.matrix(FaceCord).T).T#F
        nnFaceCord = (np.matrix(world_cam_rotM)*np.matrix(nFaceCord).T).T#W
        #    print ("nfc:\n {0}".format(nFaceCord))
        RotatedfaceCord[i] = np.float32(nnFaceCord)

    print ("RotatedfaceCord:\n {0}".format(RotatedfaceCord))

    NRotatedfaceCord = list(range(NumOfFP))
    for i in range(NumOfFP):
        I = list(range(3))
        X = RotatedfaceCord[i]
        for j in range(3):
            I[j]=X[0,j]
        NRotatedfaceCord[i]=I
    print ("NRotatedfaceCord:\n {0}".format(NRotatedfaceCord))

    NNRotatedfaceCord = list(range(NumOfFP))

    TrVec = list(range(3))
    for i in range(3):
        #TrVec[i] =  face_cam_trV[i,0] - world_cam_trV[i,0]
        #TrVec[i] =  world_cam_trV[i,0] - face_cam_trV[i,0]
        TrVec[i] =  world_cam_trV[i,0] + face_cam_trV[i,0]
        #TrVec[i] = translation_vector[i,0] - tV[i,0]
    print ("TrVec:\n {0}".format(TrVec))

    WTrMat = SRs.TvecToMat(world_cam_trV)
    FTrMat = SRs.TvecToMat(face_cam_trV)

    TNRotatedfaceCord = list(range(NumOfFP))
    for i in range(NumOfFP):
        I = NRotatedfaceCord[i]
        X = list(range(4))
        X[0]=I[0]
        X[1]=I[1]
        X[2]=I[2]
        X[3]=1
        TNRotatedfaceCord[i]=X
    #nFaceCord = (np.matrix(face_cam_rotM)*np.matrix(FaceCord).T).T#F
    TNTranslatedFC = list(range(NumOfFP))
    for i in range(NumOfFP):
        #TNTranslatedFC[i] = np.float32((np.matrix(TrMat) * np.matrix(TNRotatedfaceCord[i]).T).T)
        TNTranslatedFC[i] = np.float32((np.matrix(WTrMat) * np.matrix(TNRotatedfaceCord[i]).T).T)
    for i in range(NumOfFP):
        TNTranslatedFC[i] = np.float32((np.matrix(FTrMat) * np.matrix(TNTranslatedFC[i]).T).T)

    CorTNTranslatedFC = list(range(NumOfFP))
    for i in range(NumOfFP):
        I = list(range(3))
        X = TNTranslatedFC[i]
        for j in range(3):
            I[j]=X[0,j]
        CorTNTranslatedFC[i]=I

    #return TranslatedFC
    return CorTNTranslatedFC

def NewTranslator(world_cam_rotM, world_cam_trV, face_cam_rotM, face_cam_trV):
    ####################################

    EFC = list(range(NumOfFP))
    for i in range(NumOfFP):
        I = V.face3Dmodel[i]
        X = list(range(4))
        X[0]=I[0]
        X[1]=I[1]
        X[2]=I[2]
        X[3]=1
        EFC[i]=X

    ####################################

    WRTM = SRs.RotTrMat(world_cam_rotM, world_cam_trV)
    FRTM = SRs.RotTrMat(face_cam_rotM, face_cam_trV)
    RES = np.matrix(FRTM)*np.matrix(WRTM)

    ####################################

    RTEFC = list(range(NumOfFP))
    for i in range(NumOfFP):
      FaceCord = EFC[i]
      #    print ("fc:\n {0}".format(faceCord))
      #nFaceCord = (np.matrix(rotation_matrix)*np.matrix(faceCord).T).T#########################################np.matrix(world_cam_rotM)*
      #nnFaceCord = (np.matrix(RES)*np.matrix(FaceCord).T).T#F
      nFaceCord = (np.matrix(FRTM)*np.matrix(FaceCord).T).T#F
      nnFaceCord = (np.matrix(WRTM)*np.matrix(nFaceCord).T).T#W
      #    print ("nfc:\n {0}".format(nFaceCord))
      RTEFC[i] = np.float32(nnFaceCord)

    ###################################

    CorRTEFC = list(range(NumOfFP))
    for i in range(NumOfFP):
        I = list(range(3))
        X = RTEFC[i]
        for j in range(3):
            I[j]=X[0,j]
        CorRTEFC[i]=I
    print ("CorRTEFC:\n {0}".format(CorRTEFC))

    return CorRTEFC

#функция, которая преобразует координаты лица. Она последовательно выполняет вращение координат лица, а затем их перенос. Она умножает исходные координаты лица на матрицы вращения и затем на матрицы перемещения, на вход она получает матрицы вращения и вектора перемещения для мировой системы координат и для системы координат лица.
def AbsoluteNewTranslator(world_cam_rotM, world_cam_trV, face_cam_rotM, face_cam_trV):
    #get and extind face coordinete
    ExtendedFaceCord = list(range(NumOfFP))
    for i in range(NumOfFP):
        I = V.face3Dmodel[i]
        print (I)
        X = SRs.VecToEvec(I)
        print (X)
        ExtendedFaceCord[i] = X
    print ("ExtendedFaceCord:\n {0}".format(ExtendedFaceCord))

    # X = [[1,0,0,0],#Rot Y 180*
    #    [0,-1,0,0],
    #    [0,0,-1,0],
    #    [0,0,0,1]]

    r = Rot.from_euler('zyx', [0, 0, 0], degrees=True)#180,0,0#0,0,180#180,0,-60
    rr = r.as_matrix()
    #print ("rr:\n {0}".format(rr))
    RR = SRs.rotME(rr)
    print ("RR:\n {0}".format(RR))

    NextStepCord = ExtendedFaceCord

    # R0ExtendedFaceCord = list(range(NumOfFP))
    # RM = RR
    # for i in range(NumOfFP):
    #   FaceCord = NextStepCord[i]
    #   nFaceCord = (np.matrix(RM)*np.matrix(FaceCord).T).T#F
    #   #nFaceCord = (np.matrix(FRM.T)*np.matrix(nFaceCord).T).T#F
    #   #nFaceCord = (np.matrix(X)*np.matrix(FaceCord).T).T#F
    #   R0ExtendedFaceCord[i] = np.float32(nFaceCord)
    # NextStepCord = R0ExtendedFaceCord

    R1ExtendedFaceCord = list(range(NumOfFP))
    FRM = SRs.rotME(face_cam_rotM)
    for i in range(NumOfFP):
      FaceCord = NextStepCord[i]
      nFaceCord = (np.matrix(FRM)*np.matrix(FaceCord).T).T#F#np.linalg.inv(FRM)
      #nFaceCord = (np.matrix(FRM.T)*np.matrix(nFaceCord).T).T#F
      #nFaceCord = (np.matrix(X)*np.matrix(FaceCord).T).T#F
      R1ExtendedFaceCord[i] = np.float32(nFaceCord)
    NextStepCord = R1ExtendedFaceCord

    R2ExtendedFaceCord = list(range(NumOfFP))
    WRM = SRs.rotME(world_cam_rotM)
    for i in range(NumOfFP):
      FaceCord = NextStepCord[i]
      nFaceCord = (np.matrix(np.linalg.inv(WRM))*np.matrix(FaceCord).T).T#F
      #nFaceCord = (np.matrix(FRM.T)*np.matrix(nFaceCord).T).T#F
      #nFaceCord = (np.matrix(X)*np.matrix(FaceCord).T).T#F
      R2ExtendedFaceCord[i] = np.float32(nFaceCord)
    NextStepCord = R2ExtendedFaceCord

    T1ExtendedFaceCord = list(range(NumOfFP))
    FTM = SRs.TvecToMat(-face_cam_trV)
    for i in range(NumOfFP):
      FaceCord = NextStepCord[i]
      nFaceCord = (np.matrix(FTM)*np.matrix(FaceCord).T).T#F
      #nFaceCord = (np.matrix(FRM.T)*np.matrix(nFaceCord).T).T#F
      #nFaceCord = (np.matrix(X)*np.matrix(FaceCord).T).T#F
      T1ExtendedFaceCord[i] = np.float32(nFaceCord)
    NextStepCord = T1ExtendedFaceCord

    T2ExtendedFaceCord = list(range(NumOfFP))
    WTM = SRs.TvecToMat(world_cam_trV)
    for i in range(NumOfFP):
      FaceCord = NextStepCord[i]
      nFaceCord = (np.matrix(WTM)*np.matrix(FaceCord).T).T#F
      #nFaceCord = (np.matrix(FRM.T)*np.matrix(nFaceCord).T).T#F
      #nFaceCord = (np.matrix(X)*np.matrix(FaceCord).T).T#F
      T2ExtendedFaceCord[i] = np.float32(nFaceCord)
    NextStepCord = T2ExtendedFaceCord

    ReturnedFaceCord = SRs.NormExFC(NextStepCord, 3, 6)
    return ReturnedFaceCord

#делает то, же, что и предыдущая функция, но использует другой алгоритм, основаннный на полноразмерных матрицах 4x4
def NewFullMatrixTranslator(world_cam_rotM, world_cam_trV, face_cam_rotM, face_cam_trV):
    print("_______________________")
    ExtendedFaceCord = list(range(NumOfFP))
    for i in range(NumOfFP):
        I = V.face3Dmodel[i]
        print (I)
        X = SRs.VecToEvec(I)
        print (X)
        ExtendedFaceCord[i] = X
    print ("ExtendedFaceCord:\n {0}".format(ExtendedFaceCord))

    r = Rot.from_euler('zyx', [0, 90, 0], degrees=True)#180,0,0#0,0,180#180,0,-60
    rr = r.as_matrix()
    #print ("rr:\n {0}".format(rr))
    RR = SRs.rotME(rr)
    print ("RR:\n {0}".format(RR))

    NextStepCord = ExtendedFaceCord

    R0ExtendedFaceCord = list(range(NumOfFP))
    RM = RR
    for i in range(NumOfFP):
      FaceCord = NextStepCord[i]
      nFaceCord = (np.matrix(RM)*np.matrix(FaceCord).T).T#F
      #nFaceCord = (np.matrix(FRM.T)*np.matrix(nFaceCord).T).T#F
      #nFaceCord = (np.matrix(X)*np.matrix(FaceCord).T).T#F
      R0ExtendedFaceCord[i] = np.float32(nFaceCord)
    NextStepCord = R0ExtendedFaceCord

    FRTM = SRs.RotTrMat(face_cam_rotM, face_cam_trV)
    WRTM = SRs.RotTrMat(world_cam_rotM, world_cam_trV)
    #FWM = np.matrix(WRTM)*np.matrix(FRTM)#np.linalg.inv(WRTM)
    RRM = np.matrix(world_cam_rotM)*np.matrix(face_cam_rotM)#np.linalg.inv(world_cam_rotM)#world_cam_rotM
    NRRM = SRs.NormExFC(RRM, 3, 3)
    FWM = SRs.RTToMat(NRRM, face_cam_trV, world_cam_trV)

    CameraWorldExtendedFaceCord = list(range(NumOfFP))
    for i in range(NumOfFP):
      FaceCord = NextStepCord[i]
      nFaceCord = (np.matrix(FWM)*np.matrix(FaceCord).T).T#F
      #nFaceCord = (np.matrix(FRM.T)*np.matrix(nFaceCord).T).T#F
      #nFaceCord = (np.matrix(X)*np.matrix(FaceCord).T).T#F
      CameraWorldExtendedFaceCord[i] = np.float32(nFaceCord)
    NextStepCord = CameraWorldExtendedFaceCord

    ReturnedFaceCord = SRs.NormExFC(NextStepCord, 3, NumOfFP)
    return ReturnedFaceCord

#функция, которая реализует идею, подсказанную мне Алексеем (смотри рисунок ниже в readme)
def AlexeyIdeaTranslator(world_cam_rotM, world_cam_trV, face_cam_rotM, face_cam_trV, MC_world_cam_rotM, MC_world_cam_trV):
    ExtendedFaceCord = list(range(NumOfFP))
    for i in range(NumOfFP):
        I = V.face3Dmodel[i]
        print (I)
        X = SRs.VecToEvec(I)
        print (X)
        ExtendedFaceCord[i] = X
    print ("ExtendedFaceCord:\n {0}".format(ExtendedFaceCord))

    NextStepCord1 = ExtendedFaceCord
    NextStepCord2 = ExtendedFaceCord

    R1ExtendedFaceCord = list(range(NumOfFP))
    FRM = SRs.rotME(face_cam_rotM)
    for i in range(NumOfFP):
      FaceCord = NextStepCord1[i]
      nFaceCord = (np.matrix(FRM)*np.matrix(FaceCord).T).T#F#np.linalg.inv(FRM)
      #nFaceCord = (np.matrix(FRM.T)*np.matrix(nFaceCord).T).T#F
      #nFaceCord = (np.matrix(X)*np.matrix(FaceCord).T).T#F
      R1ExtendedFaceCord[i] = np.float32(nFaceCord)
    NextStepCord1 = R1ExtendedFaceCord

    Xcam1 = SRs.NormExFC(NextStepCord1, 3, 6)

    R2ExtendedFaceCord = list(range(NumOfFP))
    FRM = SRs.rotME(face_cam_rotM)
    WRM1 = SRs.rotME(np.linalg.inv(world_cam_rotM))
    WRM2 = SRs.rotME(MC_world_cam_rotM)
    for i in range(NumOfFP):
      FaceCord = NextStepCord2[i]
      nFaceCord = (np.matrix(FRM)*np.matrix(FaceCord).T).T#F#np.linalg.inv(FRM)
      nnFaceCord = (np.matrix(WRM1)*np.matrix(nFaceCord).T).T
      nnnFaceCord = (np.matrix(WRM2)*np.matrix(nnFaceCord).T).T
      #nFaceCord = (np.matrix(FRM.T)*np.matrix(nFaceCord).T).T#F
      #nFaceCord = (np.matrix(X)*np.matrix(FaceCord).T).T#F
      R2ExtendedFaceCord[i] = np.float32(nnnFaceCord)
    NextStepCord2 = R2ExtendedFaceCord

    Xcam2 = SRs.NormExFC(NextStepCord2, 3, 6)

    return Xcam1, Xcam2

def hopeLastRotator(face_cam_rotM, world_cam_rotM):
    ExtendedFaceCord = list(range(NumOfFP))
    for i in range(NumOfFP):
        I = V.face3Dmodel[i]
        print (I)
        X = SRs.VecToEvec(I)
        print (X)
        ExtendedFaceCord[i] = X
    print ("ExtendedFaceCord:\n {0}".format(ExtendedFaceCord))

    NextStepCord = ExtendedFaceCord

    R1ExtendedFaceCord = list(range(NumOfFP))
    FRM = SRs.rotME(face_cam_rotM)
    for i in range(NumOfFP):
        FaceCord = NextStepCord[i]
        nFaceCord = (np.matrix(FRM)*np.matrix(FaceCord).T).T#F#np.linalg.inv(FRM)
        #nFaceCord = (np.matrix(FRM.T)*np.matrix(nFaceCord).T).T#F
        #nFaceCord = (np.matrix(X)*np.matrix(FaceCord).T).T#F
        R1ExtendedFaceCord[i] = np.float32(nFaceCord)
    NextStepCord = R1ExtendedFaceCord

    R2ExtendedFaceCord = list(range(NumOfFP))
    WRM = SRs.rotME(np.linalg.inv(world_cam_rotM))
    for i in range(NumOfFP):
        FaceCord = NextStepCord[i]
        nFaceCord = (np.matrix(WRM)*np.matrix(FaceCord).T).T#F#np.linalg.inv(FRM)
        #nFaceCord = (np.matrix(FRM.T)*np.matrix(nFaceCord).T).T#F
        #nFaceCord = (np.matrix(X)*np.matrix(FaceCord).T).T#F
        R2ExtendedFaceCord[i] = np.float32(nFaceCord)
    NextStepCord = R2ExtendedFaceCord

    print ("NextStepCord:\n {0}".format(NextStepCord))
    RFC = SRs.NormExFC(NextStepCord, 3, 6)

    ###
    ExtendedFaceCord = list(range(1))
    for i in range(1):
        I = SRs.refVector()
        print (I)
        X = SRs.VecToEvec(I[0])
        print (X)
        ExtendedFaceCord[i] = X
    print ("ExtendedFaceCord:\n {0}".format(ExtendedFaceCord))

    NextStepCord = ExtendedFaceCord

    R1ExtendedFaceCord = list(range(1))
    FRM = SRs.rotME(face_cam_rotM)
    for i in range(1):
        FaceCord = NextStepCord[i]
        nFaceCord = (np.matrix(FRM)*np.matrix(FaceCord).T).T#F#np.linalg.inv(FRM)
        #nFaceCord = (np.matrix(FRM.T)*np.matrix(nFaceCord).T).T#F
        #nFaceCord = (np.matrix(X)*np.matrix(FaceCord).T).T#F
        R1ExtendedFaceCord[i] = np.float32(nFaceCord)
    NextStepCord = R1ExtendedFaceCord

    R2ExtendedFaceCord = list(range(1))
    WRM = SRs.rotME(np.linalg.inv(world_cam_rotM))
    for i in range(1):
        FaceCord = NextStepCord[i]
        nFaceCord = (np.matrix(WRM)*np.matrix(FaceCord).T).T#F#np.linalg.inv(FRM)
        #nFaceCord = (np.matrix(FRM.T)*np.matrix(nFaceCord).T).T#F
        #nFaceCord = (np.matrix(X)*np.matrix(FaceCord).T).T#F
        R2ExtendedFaceCord[i] = np.float32(nFaceCord)
    NextStepCord = R2ExtendedFaceCord

    # if signal == 1:
    #     R3ExtendedFaceCord = list(range(1))
    #     R = Rot.from_euler('xyz', [0, 0, 180])
    #     RM = R.as_matrix()
    #     ERM = SRs.rotME(RM)
    #     for i in range(1):
    #         FaceCord = NextStepCord[i]
    #         nFaceCord = (np.matrix(ERM)*np.matrix(FaceCord).T).T#F#np.linalg.inv(FRM)
    #         #nFaceCord = (np.matrix(FRM.T)*np.matrix(nFaceCord).T).T#F
    #         #nFaceCord = (np.matrix(X)*np.matrix(FaceCord).T).T#F
    #         R3ExtendedFaceCord[i] = np.float32(nFaceCord)
    #     NextStepCord = R3ExtendedFaceCord

    print ("NextStepCord:\n {0}".format(NextStepCord[0]))
    RVC = SRs.NormExFC(NextStepCord, 3, 1)
    ###

    R, rsmd = Rot.align_vectors(V.face3Dmodel, RFC)
    #R, rsmd = Rot.align_vectors(ReturnedFaceCord, Xcam2)
    print ("R :\n {0}".format(R))
    print ("rsmd :\n {0}".format(rsmd))
    #Angles = RT.rotationMatrixToEulerAngles(R)
    Angles = R.as_euler('zyx', degrees=True)
    print ("Angles :\n {0}".format(Angles))

    A = R.as_euler('xyz', degrees=True)
    print ("A :\n {0}".format(A))

    RA = Rot.from_euler('xyz', [int(A[0]), int(A[1]), int(A[2])])
    print ("RA :\n {0}".format(RA))
    RAM = RA.as_matrix()
    print ("RAM :\n {0}".format(RAM))

    # Vec3DPoint = SRs.refVector()

    RotM = RAM
    Angles = A

    #RotM = R.as_matrix()
    # print("RotM :\n {0}".format(RotM))
    # Rvector = (np.matrix(RotM)*np.matrix(Vec3DPoint).T).T
    # CRvector = SRs.NormExFC(Rvector, 3, 1)
    # print ("CRvector :\n {0}".format(CRvector))

    #return RFC, R, Angles, CRvector
    return RFC, R, Angles, RVC

