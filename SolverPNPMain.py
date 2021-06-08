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
import RotTrFuncs as RT
import SolversPNP as SP

NumOfFP = V.NumOfFPv

Mnet = cv2.dnn.readNet("yolov3_final.weights", "yolov3.cfg")
# Name custom object
Mclasses = ["MaG", "OnlyM", "OnlyG", "NMaNG"]

Mlayer_names = Mnet.getLayerNames()
Moutput_layers = [Mlayer_names[i[0] - 1] for i in Mnet.getUnconnectedOutLayers()]

#основная функция, которая работает с двумя фото, которые она получает на вход (изображение с мастер камеры(видит затылок) и изображение с слейв камеры (видит лицо)). В настояций момент реализует идею Алексея (возвращает повернутые 3d точки лица в пространстве лица для слейв камеры, мастер камеры, а так же углы поворота лица для мастер камеры)
def Photo(MCimg, SCimg):
    for i in range(4):
        cv2.circle(MCimg,(int(V.MC_image_points[i][0]),int(V.MC_image_points[i][1])),3,(0,255,0),-1)

    for i in range(4):
        cv2.circle(SCimg,(int(V.SC_image_points[i][0]),int(V.SC_image_points[i][1])),3,(0,255,0),-1)

    #WMC_cameraPosition, WMCrotM, WMCtvec, MCimg = WorldCamSolvePNP(MCimg, "imMC.jpg", default_camera_matrix, default_dist_coeffs, MC_image_points, model_points)
    #WSC_cameraPosition, WSCrotM, WSCtvec, SCimg = WorldCamSolvePNP(SCimg, "imSC.jpg", default_camera_matrix, default_dist_coeffs, SC_image_points, model_points)
    WMC_cameraPosition, WMCrotM, WMCtvec, MCimg = SP.WorldCamSolvePNP(MCimg, "imMC.jpg", V.MC_camera_matrix, V.MC_dist_coeffs, V.MC_image_points, V.model_points)
    WSC_cameraPosition, WSCrotM, WSCtvec, SCimg = SP.WorldCamSolvePNP(SCimg, "imSC.jpg", V.SC_camera_matrix, V.SC_dist_coeffs, V.SC_image_points, V.model_points)

    ###########################################################################
    #Опрашиваем мастер камеру
    # isMCViewFace = isThereFaceOnImage(MCimg)
    # isFaceFree = isTheFaceFreeFromTheBarrier(MCimg)
    # if isMCViewFace == True and isFaceFree == True:
    #     #MC видит лицо и оно свободно от барьера  
    #     success, MC_2d_face_points, MCimg = SP.FacePointReturner(MCimg, "imMC.jpg")
    #     if success == True:
    #         FMC_cameraPosition, FMCrotM, FMCtvec, MCimg = SP.FaceCamSolvePNP(MCimg, "imMC.jpg", V.MC_camera_matrix, V.MC_dist_coeffs, MC_2d_face_points, V.face3Dmodel)
    #         print ("face3Dmodel :\n {0}".format(V.face3Dmodel))
    #         print ("FSC_cameraPosition :\n {0}".format(FMC_cameraPosition))

    #         RotatedFC, R, Angles, RotatedV = RT.hopeLastRotator(FMCrotM, WMCrotM)

    #         WMCrvec = cv2.Rodrigues(WMCrotM)[0]
    #         RESimg = GeneralSuccessViz(Angles, MCimg, SCimg, RotatedV, WMCrvec, WMCtvec, V.MC_camera_matrix, V.MC_dist_coeffs)
    #         return True, RESimg

    # if isMCViewFace == True and isFaceFree == False:
    #     #MC видит лицо но оно имеет барьер
    #     WMCrvec = cv2.Rodrigues(WMCrotM)[0]
    #     RESimg = GeneralUnsuccessViz("Barrier detected!", MCimg, SCimg, WMCrvec, WMCtvec, V.MC_camera_matrix, V.MC_dist_coeffs)
    #     return True, RESimg

    # #Опрашивает слейв камеру
    # isSCViewFace = isThereFaceOnImage(SCimg)
    # isFaceFree = isTheFaceFreeFromTheBarrier(SCimg)
    # if isSCViewFace == True and isFaceFree == True:
    #     #SC видит лицо и оно свободно от барьера
    #     success, SC_2d_face_points, SCimg = SP.FacePointReturner(SCimg, "imSC.jpg")
    #     if success == True:
    #         FSC_cameraPosition, FSCrotM, FSCtvec, SCimg = SP.FaceCamSolvePNP(SCimg, "imSC.jpg", V.SC_camera_matrix, V.SC_dist_coeffs, SC_2d_face_points, V.face3Dmodel)
    #         print ("face3Dmodel :\n {0}".format(V.face3Dmodel))
    #         print ("FSC_cameraPosition :\n {0}".format(FSC_cameraPosition))

    #         RotatedFC, R, Angles, RotatedV = RT.hopeLastRotator(FSCrotM, WSCrotM)

    #         WMCrvec = cv2.Rodrigues(WMCrotM)[0]
    #         RESimg = GeneralSuccessViz(Angles, MCimg, SCimg, RotatedV, WMCrvec, WMCtvec, V.MC_camera_matrix, V.MC_dist_coeffs)
    #         return True, RESimg

    # if isSCViewFace == True and isFaceFree == False:
    #     #SC видит лицо но оно имеет барьер
    #     WMCrvec = cv2.Rodrigues(WMCrotM)[0]
    #     RESimg = GeneralUnsuccessViz("Barrier detected!", MCimg, SCimg, WMCrvec, WMCtvec, V.MC_camera_matrix, V.MC_dist_coeffs)
    #     return True, RESimg
            
    # if isMCViewFace == False and isSCViewFace == False:
    #     #В противном случае ни одна камера не видит лицо
    #     WMCrvec = cv2.Rodrigues(WMCrotM)[0]
    #     RESimg = GeneralUnsuccessViz("Can't find face!", MCimg, SCimg, WMCrvec, WMCtvec, V.MC_camera_matrix, V.MC_dist_coeffs)
    #     return True, RESimg
    ###########################################################################
    # #Опрашиваем мастер камеру
    # isMCViewFace = isThereFaceOnImage(MCimg)
    # isFaceFree = isTheFaceFreeFromTheBarrier(MCimg)###
    # #if isFaceFree == True:###
    # if isMCViewFace == True:
    #     #MC видит лицо
    #     #isFaceFree = isTheFaceFreeFromTheBarrier(MCimg)
    #     if isFaceFree == True:
    #         #MC видит лицо и оно свободно от барьера
    #         success, MC_2d_face_points, MCimg = SP.FacePointReturner(MCimg, "imMC.jpg")
    #         if success == True:
    #             FMC_cameraPosition, FMCrotM, FMCtvec, MCimg = SP.FaceCamSolvePNP(MCimg, "imMC.jpg", V.MC_camera_matrix, V.MC_dist_coeffs, MC_2d_face_points, V.face3Dmodel)
    #             print ("face3Dmodel :\n {0}".format(V.face3Dmodel))
    #             print ("FSC_cameraPosition :\n {0}".format(FMC_cameraPosition))

    #             RotatedFC, R, Angles, RotatedV = RT.hopeLastRotator(FMCrotM, WMCrotM)

    #             WMCrvec = cv2.Rodrigues(WMCrotM)[0]
    #             RESimg = GeneralSuccessViz(Angles, MCimg, SCimg, RotatedV, WMCrvec, WMCtvec, V.MC_camera_matrix, V.MC_dist_coeffs)
    #             return True, RESimg
    #     else:
    #         #MC видит лицо но оно имеет барьер
    #         WMCrvec = cv2.Rodrigues(WMCrotM)[0]
    #         RESimg = GeneralUnsuccessViz("Barrier detected!", MCimg, SCimg, WMCrvec, WMCtvec, V.MC_camera_matrix, V.MC_dist_coeffs)
    #         return True, RESimg
    # else:
    #     #MC видит лицо но оно имеет барьер###
    #     WMCrvec = cv2.Rodrigues(WMCrotM)[0]###
    #     RESimg = GeneralUnsuccessViz("Camera can't find face!", MCimg, SCimg, WMCrvec, WMCtvec, V.MC_camera_matrix, V.MC_dist_coeffs)###
    #     return True, RESimg###

    # #Опрашивает слейв камеру
    # isSCViewFace = isThereFaceOnImage(SCimg)
    # isFaceFree = isTheFaceFreeFromTheBarrier(SCimg)###
    # #if isFaceFree == True:###
    # if isSCViewFace == True:
    #     #SC видит лицо
    #     # isFaceFree = isTheFaceFreeFromTheBarrier(SCimg)
    #     if isFaceFree == True:
    #         #SC видит лицо и оно свободно от барьера
    #         success, SC_2d_face_points, SCimg = SP.FacePointReturner(SCimg, "imSC.jpg")
    #         if success == True:
    #             FSC_cameraPosition, FSCrotM, FSCtvec, SCimg = SP.FaceCamSolvePNP(SCimg, "imSC.jpg", V.SC_camera_matrix, V.SC_dist_coeffs, SC_2d_face_points, V.face3Dmodel)
    #             print ("face3Dmodel :\n {0}".format(V.face3Dmodel))
    #             print ("FSC_cameraPosition :\n {0}".format(FSC_cameraPosition))

    #             RotatedFC, R, Angles, RotatedV = RT.hopeLastRotator(FSCrotM, WSCrotM)

    #             WMCrvec = cv2.Rodrigues(WMCrotM)[0]
    #             RESimg = GeneralSuccessViz(Angles, MCimg, SCimg, RotatedV, WMCrvec, WMCtvec, V.MC_camera_matrix, V.MC_dist_coeffs)
    #             return True, RESimg
    #     else:
    #         #SC видит лицо но оно имеет барьер
    #         WMCrvec = cv2.Rodrigues(WMCrotM)[0]
    #         RESimg = GeneralUnsuccessViz("Barrier detected!", MCimg, SCimg, WMCrvec, WMCtvec, V.MC_camera_matrix, V.MC_dist_coeffs)
    #         return True, RESimg
    # else:###
    #     #SC видит лицо но оно имеет барьер###
    #     WMCrvec = cv2.Rodrigues(WMCrotM)[0]###
    #     RESimg = GeneralUnsuccessViz("Barrier detected!", MCimg, SCimg, WMCrvec, WMCtvec, V.MC_camera_matrix, V.MC_dist_coeffs)###
    #     return True, RESimg###
            
    # if isMCViewFace == False and isSCViewFace == False:
    #     #В противном случае ни одна камера не видит лицо
    #     WMCrvec = cv2.Rodrigues(WMCrotM)[0]
    #     RESimg = GeneralUnsuccessViz("Can't find face!", MCimg, SCimg, WMCrvec, WMCtvec, V.MC_camera_matrix, V.MC_dist_coeffs)
    #     return True, RESimg
    ###########################################################################
    #Опрашиваем мастер камеру
    isMCViewFace = isThereFaceOnImage(MCimg)
    if isMCViewFace == True:
        #MC видит лицо
        isFaceFree = isTheFaceFreeFromTheBarrier(MCimg)
        if isFaceFree == True:
            #MC видит лицо и оно свободно от барьера
            success, MC_2d_face_points, MCimg = SP.FacePointReturner(MCimg, "imMC.jpg")
            if success == True:
                FMC_cameraPosition, FMCrotM, FMCtvec, MCimg = SP.FaceCamSolvePNP(MCimg, "imMC.jpg", V.MC_camera_matrix, V.MC_dist_coeffs, MC_2d_face_points, V.face3Dmodel)
                print ("face3Dmodel :\n {0}".format(V.face3Dmodel))
                print ("FSC_cameraPosition :\n {0}".format(FMC_cameraPosition))

                RotatedFC, R, Angles, RotatedV = RT.hopeLastRotator(FMCrotM, WMCrotM)

                WMCrvec = cv2.Rodrigues(WMCrotM)[0]
                RESimg = GeneralSuccessViz(Angles, MCimg, SCimg, RotatedV, WMCrvec, WMCtvec, V.MC_camera_matrix, V.MC_dist_coeffs)
                return True, RESimg
        else:
            #MC видит лицо но оно имеет барьер
            WMCrvec = cv2.Rodrigues(WMCrotM)[0]
            RESimg = GeneralUnsuccessViz("Barrier detected!", MCimg, SCimg, WMCrvec, WMCtvec, V.MC_camera_matrix, V.MC_dist_coeffs)
            return True, RESimg

    #Опрашивает слейв камеру
    isSCViewFace = isThereFaceOnImage(SCimg)
    if isSCViewFace == True:
        #SC видит лицо
        isFaceFree = isTheFaceFreeFromTheBarrier(SCimg)
        if isFaceFree == True:
            #SC видит лицо и оно свободно от барьера
            success, SC_2d_face_points, SCimg = SP.FacePointReturner(SCimg, "imSC.jpg")
            if success == True:
                FSC_cameraPosition, FSCrotM, FSCtvec, SCimg = SP.FaceCamSolvePNP(SCimg, "imSC.jpg", V.SC_camera_matrix, V.SC_dist_coeffs, SC_2d_face_points, V.face3Dmodel)
                print ("face3Dmodel :\n {0}".format(V.face3Dmodel))
                print ("FSC_cameraPosition :\n {0}".format(FSC_cameraPosition))

                RotatedFC, R, Angles, RotatedV = RT.hopeLastRotator(FSCrotM, WSCrotM)

                WMCrvec = cv2.Rodrigues(WMCrotM)[0]
                RESimg = GeneralSuccessViz(Angles, MCimg, SCimg, RotatedV, WMCrvec, WMCtvec, V.MC_camera_matrix, V.MC_dist_coeffs)
                return True, RESimg
        else:
            #SC видит лицо но оно имеет барьер
            WMCrvec = cv2.Rodrigues(WMCrotM)[0]
            RESimg = GeneralUnsuccessViz("Barrier detected!", MCimg, SCimg, WMCrvec, WMCtvec, V.MC_camera_matrix, V.MC_dist_coeffs)
            return True, RESimg
            
    if isMCViewFace == False and isSCViewFace == False:
        #В противном случае ни одна камера не видит лицо
        WMCrvec = cv2.Rodrigues(WMCrotM)[0]
        RESimg = GeneralUnsuccessViz("Can't find face!", MCimg, SCimg, WMCrvec, WMCtvec, V.MC_camera_matrix, V.MC_dist_coeffs)
        return True, RESimg
    ###########################################################################

    # ########

    # success, SC_2d_face_points, SCimg = SP.FacePointReturner(SCimg, "imSC.jpg")
    # if success == True:
    #     FSC_cameraPosition, FSCrotM, FSCtvec, SCimg = SP.FaceCamSolvePNP(SCimg, "imSC.jpg", V.SC_camera_matrix, V.SC_dist_coeffs, SC_2d_face_points, V.face3Dmodel)
    #     print ("face3Dmodel :\n {0}".format(V.face3Dmodel))
    #     print ("FSC_cameraPosition :\n {0}".format(FSC_cameraPosition))

    #     SCTranslatedFC = RT.AbsoluteNewTranslator(WSCrotM, WSCtvec, FSCrotM, FSCtvec)

    #     NFMSCTranslatedFC = RT.NewFullMatrixTranslator(WSCrotM, WSCtvec, FSCrotM, FSCtvec)

    #     Xcam1, Xcam2 = RT.AlexeyIdeaTranslator(WSCrotM, WSCtvec, FSCrotM, FSCtvec, WMCrotM, WMCtvec)
    #     print ("Xcam1 :\n {0}".format(Xcam1))
    #     print ("Xcam2 :\n {0}".format(Xcam2))

    #     #checing
    #     WSCrvec = cv2.Rodrigues(WSCrotM)[0]
    #     WMCrvec = cv2.Rodrigues(WMCrotM)[0]
    #     FSCrvec = cv2.Rodrigues(FSCrotM)[0]
    #     WMCrvec = cv2.Rodrigues(WMCrotM)[0]

    #     R, rsmd = Rot.align_vectors(V.face3Dmodel, Xcam2)
    #     #R, rsmd = Rot.align_vectors(ReturnedFaceCord, Xcam2)
    #     print ("R :\n {0}".format(R))
    #     print ("rsmd :\n {0}".format(rsmd))
    #     #Angles = RT.rotationMatrixToEulerAngles(R)
    #     Angles = R.as_euler('zyx', degrees=True)
    #     print ("Angles :\n {0}".format(Angles))

    #     Vec3DPoint = SRs.refVector()
    #     print ("Vec3DPoint :\n {0}".format(Vec3DPoint))
    #     RotM = R.as_matrix()
    #     print ("RotM :\n {0}".format(RotM))
    #     MCimgWV = VectorViz(Angles, MCimg, Vec3DPoint, RotM, WMCrvec, WMCtvec, V.MC_camera_matrix, V.MC_dist_coeffs)

    #     cv2.imshow("MCimgWV", MCimgWV)

    #     return True, Xcam1, Xcam2, Angles, MCimgWV

    # else:
    #     A = list(range(6))
    #     B = list(range(6))
    #     C = list(range(6))
    #     return False, A, B, C, MCimg

    # ########

    #ЭТО ЕЩЕ МОЖЕТ ПОНАДОБИТЬСЯ ДЛЯ 3d ВИЗУАЛИЗАЦИИ

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # #SCTranslatedFC
    # #face3Dmodel
    # d3PlotPoints = SCTranslatedFC
    # print ("d3PlotPoints :\n {0}".format(d3PlotPoints))

    # x =[d3PlotPoints[0][0],d3PlotPoints[1][0],d3PlotPoints[2][0],d3PlotPoints[3][0],d3PlotPoints[4][0],d3PlotPoints[5][0]]
    # y =[d3PlotPoints[0][1],d3PlotPoints[1][1],d3PlotPoints[2][1],d3PlotPoints[3][1],d3PlotPoints[4][1],d3PlotPoints[5][1]]
    # z =[d3PlotPoints[0][2],d3PlotPoints[1][2],d3PlotPoints[2][2],d3PlotPoints[3][2],d3PlotPoints[4][2],d3PlotPoints[5][2]]
    # ax.scatter(x, y, z, c='red', marker='o')

    # # d3PlotPoints = NFMSCTranslatedFC
    # # print ("d3PlotPoints :\n {0}".format(d3PlotPoints))

    # # x =[d3PlotPoints[0][0],d3PlotPoints[1][0],d3PlotPoints[2][0],d3PlotPoints[3][0],d3PlotPoints[4][0],d3PlotPoints[5][0]]
    # # y =[d3PlotPoints[0][1],d3PlotPoints[1][1],d3PlotPoints[2][1],d3PlotPoints[3][1],d3PlotPoints[4][1],d3PlotPoints[5][1]]
    # # z =[d3PlotPoints[0][2],d3PlotPoints[1][2],d3PlotPoints[2][2],d3PlotPoints[3][2],d3PlotPoints[4][2],d3PlotPoints[5][2]]
    # #ax.scatter(x, y, z, c='purple', marker='o')

    # d3PlotPoints = Xcam1
    # print ("d3PlotPoints :\n {0}".format(d3PlotPoints))

    # x =[d3PlotPoints[0][0],d3PlotPoints[1][0],d3PlotPoints[2][0],d3PlotPoints[3][0],d3PlotPoints[4][0],d3PlotPoints[5][0]]
    # y =[d3PlotPoints[0][1],d3PlotPoints[1][1],d3PlotPoints[2][1],d3PlotPoints[3][1],d3PlotPoints[4][1],d3PlotPoints[5][1]]
    # z =[d3PlotPoints[0][2],d3PlotPoints[1][2],d3PlotPoints[2][2],d3PlotPoints[3][2],d3PlotPoints[4][2],d3PlotPoints[5][2]]
    # ax.scatter(x, y, z, c='purple', marker='o')

    # d3PlotPoints = Xcam2
    # print ("d3PlotPoints :\n {0}".format(d3PlotPoints))

    # x =[d3PlotPoints[0][0],d3PlotPoints[1][0],d3PlotPoints[2][0],d3PlotPoints[3][0],d3PlotPoints[4][0],d3PlotPoints[5][0]]
    # y =[d3PlotPoints[0][1],d3PlotPoints[1][1],d3PlotPoints[2][1],d3PlotPoints[3][1],d3PlotPoints[4][1],d3PlotPoints[5][1]]
    # z =[d3PlotPoints[0][2],d3PlotPoints[1][2],d3PlotPoints[2][2],d3PlotPoints[3][2],d3PlotPoints[4][2],d3PlotPoints[5][2]]
    # ax.scatter(x, y, z, c='black', marker='o')

    # CamPlotPoints = WMC_cameraPosition
    # print ("CamPlotPoints :\n {0}".format(CamPlotPoints))

    # x0 = [CamPlotPoints[0,0]]
    # y0 = [CamPlotPoints[1,0]]
    # z0 = [CamPlotPoints[2,0]]
    # ax.scatter(x0, y0, z0, c='green', marker='o')

    # CamPlotPoints = WSC_cameraPosition
    # print ("CamPlotPoints :\n {0}".format(CamPlotPoints))

    # x1 = [CamPlotPoints[0,0]]
    # y1 = [CamPlotPoints[1,0]]
    # z1 = [CamPlotPoints[2,0]]
    # ax.scatter(x1, y1, z1, c='yellow', marker='o')

    # CamPlotPoints = FSC_cameraPosition
    # print ("CamPlotPoints :\n {0}".format(CamPlotPoints))

    # x1 = [CamPlotPoints[0,0]]
    # y1 = [CamPlotPoints[1,0]]
    # z1 = [CamPlotPoints[2,0]]
    # ax.scatter(x1, y1, z1, c='orange', marker='o')

    # CamPlotPoints = FSCtvec
    # print ("CamPlotPoints :\n {0}".format(CamPlotPoints))

    # x1 = [CamPlotPoints[0,0]]
    # y1 = [CamPlotPoints[1,0]]
    # z1 = [CamPlotPoints[2,0]]
    # #ax.scatter(x1, y1, z1, c='black', marker='o')

    # CamPlotPoints = WSCtvec
    # print ("CamPlotPoints :\n {0}".format(CamPlotPoints))

    # x1 = [CamPlotPoints[0,0]]
    # y1 = [CamPlotPoints[1,0]]
    # z1 = [CamPlotPoints[2,0]]
    # #ax.scatter(x1, y1, z1, c='violet', marker='o')

    # x2 = [model_points[0][0], model_points[1][0], model_points[2][0], model_points[3][0]]
    # y2 = [model_points[0][1], model_points[1][1], model_points[2][1], model_points[3][1]]
    # z2 = [model_points[0][2], model_points[1][2], model_points[2][2], model_points[3][2]]
    # ax.scatter(x2, y2, z2, c='blue', marker='o')

    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    # plt.show()
    #return Xcam1, Xcam2, Angles, MCimgWV

def isThereFaceOnImage(img):
    # faces = face_recognition.face_locations(img, model="cnn")
    # Конвертирование изображения в черно-белое
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц и построение прямоугольного контура
    faces = V.detector(gimg)

    if not faces:
        return False
    else:
        return True

def isTheFaceFreeFromTheBarrier(img):
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    #####
    Mnet.setInput(blob)
    Mouts = Mnet.forward(Moutput_layers)

    for Mout in Mouts:
        for Mdetection in Mout:
            Mscores = Mdetection[5:]
            Mclass_id = np.argmax(Mscores)
            Mconfidence = Mscores[Mclass_id]
            if Mconfidence > 0.7:
                # Object detected
                #print("M "+str(Mclass_id))
                if Mclass_id == 3:
                    #print("true")
                    return True
                else:
                    #print("false")
                    return False

#данная функция визуализирует положение точек лица и камеры в 3-ех мерном пространстве
def FaceInWorldViz(OriginFacePoints, facePoints, CPP):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    #SCTranslatedFC
    #face3Dmodel
    d3PlotPoints = facePoints,
    print ("d3PlotPoints :\n {0}".format(d3PlotPoints))

    x =[d3PlotPoints[0][0][0],d3PlotPoints[0][1][0],d3PlotPoints[0][2][0],d3PlotPoints[0][3][0],d3PlotPoints[0][4][0],d3PlotPoints[0][5][0]]
    y =[d3PlotPoints[0][0][1],d3PlotPoints[0][1][1],d3PlotPoints[0][2][1],d3PlotPoints[0][3][1],d3PlotPoints[0][4][1],d3PlotPoints[0][5][1]]
    z =[d3PlotPoints[0][0][2],d3PlotPoints[0][1][2],d3PlotPoints[0][2][2],d3PlotPoints[0][3][2],d3PlotPoints[0][4][2],d3PlotPoints[0][5][2]]
    ax.scatter(x, y, z, c='red', marker='o')

    d3PlotPoints = OriginFacePoints,
    print ("d3PlotPoints :\n {0}".format(d3PlotPoints))

    x =[d3PlotPoints[0][0][0],d3PlotPoints[0][1][0],d3PlotPoints[0][2][0],d3PlotPoints[0][3][0],d3PlotPoints[0][4][0],d3PlotPoints[0][5][0]]
    y =[d3PlotPoints[0][0][1],d3PlotPoints[0][1][1],d3PlotPoints[0][2][1],d3PlotPoints[0][3][1],d3PlotPoints[0][4][1],d3PlotPoints[0][5][1]]
    z =[d3PlotPoints[0][0][2],d3PlotPoints[0][1][2],d3PlotPoints[0][2][2],d3PlotPoints[0][3][2],d3PlotPoints[0][4][2],d3PlotPoints[0][5][2]]
    ax.scatter(x, y, z, c='green', marker='o')

    x2 = [V.model_points[0][0], V.model_points[1][0], V.model_points[2][0], V.model_points[3][0]]
    y2 = [V.model_points[0][1], V.model_points[1][1], V.model_points[2][1], V.model_points[3][1]]
    z2 = [V.model_points[0][2], V.model_points[1][2], V.model_points[2][2], V.model_points[3][2]]
    ax.scatter(x2, y2, z2, c='blue', marker='o')

    CamPlotPoints = CPP
    print ("CamPlotPoints :\n {0}".format(CamPlotPoints))

    x1 = [CamPlotPoints[0,0]]
    y1 = [CamPlotPoints[1,0]]
    z1 = [CamPlotPoints[2,0]]
    ax.scatter(x1, y1, z1, c='orange', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def GeneralSuccessViz(Angles, MCimg, SCimg, RotatedVector, WMCrvec, WMCtvec, MC_camera_matrix, MC_dist_coeffs):
    # cv2.putText(MCimg, "Z: " + str(int(Angles[0])), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    # cv2.putText(MCimg, "Y: " + str(int(Angles[1])), (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    # cv2.putText(MCimg, "X: " + str(int(Angles[2])), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

    cv2.putText(MCimg, "X: " + str(int(Angles[0])), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    cv2.putText(MCimg, "Y: " + str(int(Angles[1])), (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(MCimg, "Z: " + str(int(Angles[2])), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    
    UUU_end_point, jac = cv2.projectPoints(np.float32(SRs.refUUUVector()), WMCrvec, WMCtvec, MC_camera_matrix, MC_dist_coeffs)
    cv2.circle(MCimg,(int(UUU_end_point[0][0][0]),int(UUU_end_point[0][0][1])),3,(255,0,0),-1)
    p1 = (int(V.MC_image_points[0][0]),int(V.MC_image_points[0][1]))
    p2 = (int(UUU_end_point[0][0][0]),int(UUU_end_point[0][0][1]))
    cv2.line(MCimg, p1, p2, (255,0,0), 2)

    VVV_end_point, jac = cv2.projectPoints(np.float32(SRs.refVVVVector()), WMCrvec, WMCtvec, MC_camera_matrix, MC_dist_coeffs)
    cv2.circle(MCimg,(int(VVV_end_point[0][0][0]),int(VVV_end_point[0][0][1])),3,(0,255,0),-1)
    p1 = (int(V.MC_image_points[0][0]),int(V.MC_image_points[0][1]))
    p2 = (int(VVV_end_point[0][0][0]),int(VVV_end_point[0][0][1]))
    cv2.line(MCimg, p1, p2, (0,255,0), 2)

    WWW_end_point, jac = cv2.projectPoints(np.float32(SRs.refWWWVector()), WMCrvec, WMCtvec, MC_camera_matrix, MC_dist_coeffs)
    cv2.circle(MCimg,(int(WWW_end_point[0][0][0]),int(WWW_end_point[0][0][1])),3,(0,0,255),-1)
    p1 = (int(V.MC_image_points[0][0]),int(V.MC_image_points[0][1]))
    p2 = (int(WWW_end_point[0][0][0]),int(WWW_end_point[0][0][1]))
    cv2.line(MCimg, p1, p2, (0,0,255), 2)

    Vec_end_point, jac = cv2.projectPoints(np.float32(RotatedVector), WMCrvec, WMCtvec, MC_camera_matrix, MC_dist_coeffs)
    cv2.circle(MCimg,(int(Vec_end_point[0][0][0]),int(Vec_end_point[0][0][1])),3,(0,0,0),-1)
    p1 = (int(V.MC_image_points[0][0]),int(V.MC_image_points[0][1]))
    p2 = (int(Vec_end_point[0][0][0]),int(Vec_end_point[0][0][1]))
    cv2.line(MCimg, p1, p2, (0,0,0), 2)

    Cimg = np.concatenate((MCimg, SCimg), axis=1)
    return Cimg

def GeneralUnsuccessViz(Message, MCimg, SCimg, WMCrvec, WMCtvec, MC_camera_matrix, MC_dist_coeffs):
    cv2.putText(MCimg, Message, (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    # cv2.putText(MCimg, "Z: ?", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    # cv2.putText(MCimg, "Y: ?", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    # cv2.putText(MCimg, "X: ?", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

    cv2.putText(MCimg, "X: ?", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    cv2.putText(MCimg, "Y: ?", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(MCimg, "Z: ?", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    UUU_end_point, jac = cv2.projectPoints(np.float32(SRs.refUUUVector()), WMCrvec, WMCtvec, MC_camera_matrix, MC_dist_coeffs)
    cv2.circle(MCimg,(int(UUU_end_point[0][0][0]),int(UUU_end_point[0][0][1])),3,(255,0,0),-1)
    p1 = (int(V.MC_image_points[0][0]),int(V.MC_image_points[0][1]))
    p2 = (int(UUU_end_point[0][0][0]),int(UUU_end_point[0][0][1]))
    cv2.line(MCimg, p1, p2, (255,0,0), 2)

    VVV_end_point, jac = cv2.projectPoints(np.float32(SRs.refVVVVector()), WMCrvec, WMCtvec, MC_camera_matrix, MC_dist_coeffs)
    cv2.circle(MCimg,(int(VVV_end_point[0][0][0]),int(VVV_end_point[0][0][1])),3,(0,255,0),-1)
    p1 = (int(V.MC_image_points[0][0]),int(V.MC_image_points[0][1]))
    p2 = (int(VVV_end_point[0][0][0]),int(VVV_end_point[0][0][1]))
    cv2.line(MCimg, p1, p2, (0,255,0), 2)

    WWW_end_point, jac = cv2.projectPoints(np.float32(SRs.refWWWVector()), WMCrvec, WMCtvec, MC_camera_matrix, MC_dist_coeffs)
    cv2.circle(MCimg,(int(WWW_end_point[0][0][0]),int(WWW_end_point[0][0][1])),3,(0,0,255),-1)
    p1 = (int(V.MC_image_points[0][0]),int(V.MC_image_points[0][1]))
    p2 = (int(WWW_end_point[0][0][0]),int(WWW_end_point[0][0][1]))
    cv2.line(MCimg, p1, p2, (0,0,255), 2)

    Cimg = np.concatenate((MCimg, SCimg), axis=1)
    return Cimg

# визуализирует вектор в 3-ех мерном пространстве
def VectorViz(Angles, MCimg, Vec3DPoint, RotM, WMCrvec, WMCtvec, MC_camera_matrix, MC_dist_coeffs):
    Rvector = (np.matrix(RotM)*np.matrix(Vec3DPoint).T).T
    CRvector = SRs.NormExFC(Rvector, 3, 1)
    print ("CRvector :\n {0}".format(CRvector))

    Vec_end_point, jac = cv2.projectPoints(np.float32(CRvector), WMCrvec, WMCtvec, MC_camera_matrix, MC_dist_coeffs)

    cv2.circle(MCimg,(int(Vec_end_point[0][0][0]),int(Vec_end_point[0][0][1])),3,(0,0,255),-1)

    cv2.putText(MCimg, "Z: " + str(Angles[0]), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    cv2.putText(MCimg, "Y: " + str(Angles[1]), (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    cv2.putText(MCimg, "X: " + str(Angles[2]), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

    p1 = (int(V.MC_image_points[0][0]),int(V.MC_image_points[0][1]))
    p2 = (int(Vec_end_point[0][0][0]),int(Vec_end_point[0][0][1]))
 
    cv2.line(MCimg, p1, p2, (255,0,0), 2)

    return MCimg

# то же самое, что и функция Photo() только для двух видеопотоков
def Video(MCvideoStream, SCvideoStream):
    MCcap = cv2.VideoCapture(MCvideoStream)
    SCcap = cv2.VideoCapture(SCvideoStream)

    if (MCcap.isOpened() == False or SCcap.isOpened() == False):
        print("Unable to read camera feed")

    frame_width = int(MCcap.get(3)*2)
    frame_height = int(MCcap.get(4))

    out = cv2.VideoWriter('RESoutpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    while(True):
        MCret, MCframe = MCcap.read()
        SCret, SCframe = SCcap.read()
 
        if MCret == True and SCret == True:

            #Success, Xcam1, Xcam2, Angles, MCimgWV = Photo(MCframe, SCframe)
            Success, RES = Photo(MCframe, SCframe)
     
            # Write the frame into the file 'output.avi'
            if Success == True:
                out.write(RES)
                cv2.imshow("RES", RES)
 
            # Display the resulting frame   
            #cv2.imshow('frame',frame)
 
            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
 
        # Break the loop
        else:
            break 
 
    # When everything done, release the video capture and video write objects
    MCcap.release()
    SCcap.release()
    out.release()
 
    # Closes all the frames
    cv2.destroyAllWindows()





def Main():
	# MCimg = cv2.imread("MC.jpg")
    # SCimg = cv2.imread("SC.jpg")

    # MCvideoStream = "SMArchiv/MCoutpy.avi"
    # SCvideoStream = "SMArchiv/SCoutpy.avi"
    MCvideoStream = "MCoutpy.avi"
    SCvideoStream = "SCoutpy.avi"

    MCimg = cv2.imread("MSZapas/MC1.jpg")#1
    SCimg = cv2.imread("MSZapas/SC1.jpg")#1

    Video(MCvideoStream, SCvideoStream)
    #Photo(MCimg, SCimg)
    #FaceInWorldViz()

Main()