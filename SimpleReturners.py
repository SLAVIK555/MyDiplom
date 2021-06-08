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

#возвращает 3d координаты виртуальной камеры
def refVirtCam3DModel():
    modelPoints = [[1.0, 0.0, 0.0]]#0

    return np.array(modelPoints, dtype=np.float64)

#возвращает 3d координаты вектора
def refVector():
    modelPoints = [[10.0, 0.0, 0.0]]#0[[0.0, -10.0, 0.0]]

    return np.array(modelPoints, dtype=np.float64)

def refUUUVector():
    modelPoints = [[10.0, 0.0, 0.0]]#0[[0.0, -10.0, 0.0]]

    return np.array(modelPoints, dtype=np.float64)

def refVVVVector():
    modelPoints = [[0.0, 10.0, 0.0]]#0[[0.0, -10.0, 0.0]]

    return np.array(modelPoints, dtype=np.float64)

def refWWWVector():
    modelPoints = [[0.0, 0.0, 10.0]]#0[[0.0, -10.0, 0.0]]

    return np.array(modelPoints, dtype=np.float64)

# def ref3DModel():
#     modelPoints = [[0.0, 0.0, 0.0],#0
#                    [2.0, 0.0, -8.0],#1-##7.0-----------------------
#                    [4.0, 5.0, 4.5],#2-
#                    [4.0, -5.0, 4.5],#3-
#                    [1.5, 2.8, -3.5],#4-##0.25
#                    [1.5, -2.8, -3.5]]#5-
#     return np.array(modelPoints, dtype=np.float64)

# def ref3DModel():
#     modelPoints = [[0.0, 0.0, 0.0],#0
#                    [-2.0, 0.0, -8.0],#1-##7.0-----------------------
#                    [-4.0, -5.0, 4.5],#2-
#                    [-4.0, 5.0, 4.5],#3-
#                    [-1.5, -2.8, -3.5],#4-##0.25
#                    [-1.5, 2.8, -3.5]]#5-
#     return np.array(modelPoints, dtype=np.float64)

def ref3DModel():
    modelPoints = [[0.0, -5.0, 4.0],#0
                   [0.0, -4.0, 0.0],#1-##7.0-----------------------
                   [0.0, -2.0, -4.0],#2-
                   [0.0, 2.0, -4.0],#3-
                   [0.0, 4.0, 0.0],#4-##0.25
                   [0.0, 5.0, 4.0]]#5-
    return np.array(modelPoints, dtype=np.float64)

#возвращает 3d координаты точек лица
# def ref3DModel():
#     modelPoints = [[0.0, 5.0, 4.0],#00.0, -5.0, 4.0
#                    [0.0, 4.0, 0.0],#1-##7.0-----------------------
#                    [0.0, 2.0, -4.0],#2-
#                    [0.0, -2.0, -4.0],#3-
#                    [0.0, -4.0, 0.0],#4-##0.25
#                    [0.0, -5.0, 4.0]]#5-
#     return np.array(modelPoints, dtype=np.float64)

# def ref2dImagePoints(shape):
#     imagePoints = [[shape.part(30).x, shape.part(30).y],#0
#                    [shape.part(8).x, shape.part(8).y],#1
#                    [shape.part(36).x, shape.part(36).y],#2
#                    [shape.part(45).x, shape.part(45).y],#3
#                    [shape.part(48).x, shape.part(48).y],#4
#                    [shape.part(54).x, shape.part(54).y]]#5
#     return np.array(imagePoints, dtype=np.float64)

#возвращает нужные лендмарки, соответствующие 3d точкам из предыдущей функции (смотри катринку ниже в readme)
def ref2dImagePoints(shape):
    imagePoints = [[shape.part(0).x, shape.part(0).y],#0
                   [shape.part(3).x, shape.part(3).y],#1
                   [shape.part(6).x, shape.part(6).y],#2
                   [shape.part(10).x, shape.part(10).y],#3
                   [shape.part(13).x, shape.part(13).y],#4
                   [shape.part(16).x, shape.part(16).y]]#5
    return np.array(imagePoints, dtype=np.float64)

#преобразует вектор перевода в матрицу перевода
def TvecToMat(tvec):
    TVM = [[1,0,0,tvec[0]],
           [0,1,0,tvec[1]],
           [0,0,1,tvec[2]],
           [0,0,0,1]]
    return np.array(TVM, dtype=np.float64)

#преобразует сумму из двух векторов перевода в матрицу перевода
def TvecSToMat(tvec1, tvec2):
    TVM = [[1,0,0,tvec1[0]+tvec2[0]],
           [0,1,0,tvec1[1]+tvec2[1]],
           [0,0,1,tvec1[2]+tvec2[2]],
           [0,0,0,1]]
    return np.array(TVM, dtype=np.float64)

#преобразует матрицу вращения и вектор перевода в матрицу вращения и перевода
def RotTrMat(rotM, tvec):
    TVM = [[rotM[0][0],rotM[0][1],rotM[0][2],tvec[0]],
           [rotM[1][0],rotM[1][1],rotM[1][2],tvec[1]],
           [rotM[2][0],rotM[2][1],rotM[2][2],tvec[2]],
           [0,0,0,1]]
    return np.array(TVM, dtype=np.float64)

#преобразует матрицу вращения и сумму двух векторов перевода в матрицу вращения и перевода
def RTToMat(rotM, tvec0, tvec1):
    TVM = [[rotM[0][0],rotM[0][1],rotM[0][2],tvec0[0]+tvec1[0]],
           [rotM[1][0],rotM[1][1],rotM[1][2],tvec0[1]+tvec1[1]],
           [rotM[2][0],rotM[2][1],rotM[2][2],tvec0[1]+tvec1[1]],
           [0,0,0,1]]
    return np.array(TVM, dtype=np.float64)

# def TranslationMatrix(cam_rotM, world_cam_trV, face_cam_trV):
#     TM = [[cam_rotM[0][0],cam_rotM[0][1],cam_rotM[0][2],world_cam_trV[0]+face_cam_trV[0]],
#            [cam_rotM[1][0],cam_rotM[1][1],cam_rotM[1][2],world_cam_trV[1]+face_cam_trV[1]],
#            [cam_rotM[2][0],cam_rotM[2][1],cam_rotM[2][2],world_cam_trV[2]+face_cam_trV[2]],
#            [0,0,0,1]]
#     return np.array(TM, dtype=np.float64)

#расширяет матрицу вращения с 3x3 до 4x4
def rotME(RotM):
  M = [[RotM[0][0],RotM[0][1],RotM[0][2],0],
       [RotM[1][0],RotM[1][1],RotM[1][2],0],
       [RotM[2][0],RotM[2][1],RotM[2][2],0],
       [0,0,0,1]]
  return np.array(M, dtype=np.float64)

#сужает матрицу вращения до размера 3x3 посредством обрезания
def EmmatToMat(Emat):
  M = [[Emat[0][0],Emat[0][1],Emat[0][2]],
       [Emat[1][0],Emat[1][1],Emat[1][2]],
       [Emat[2][0],Emat[2][1],Emat[2][2]]]
  return np.array(M, dtype=np.float64)

#расширяет вектор вращения с размера 3x3 до 4x4 добавление единицы в конце
def VecToEvec(Vec):
  M = [Vec[0],
       Vec[1],
       Vec[2],
       1]
  return np.array(M, dtype=np.float64)

#преобразует объект типа список np.matrix-ов в объект типа список списков, lsize - размер списка, oDim - размер списков, которые являются элементами главного списка
def NormExFC(FC, oDim, lsize):
  NormFC = list(range(lsize))
  for i in range(lsize):
      I = list(range(oDim))
      X = FC[i]
      for j in range(oDim):
          #print ("X[0,j]:\n {0}".format(X[0][j]))
          I[j]=X[0,j]
      NormFC[i]=I
  return NormFC

#преобразует вектор размера 1x3 в диагональную матрицу размера 3x3
def OneDimToMat(ODM):
  M = [[ODM[0,0],0,0],
       [0,ODM[0,1],0],
       [0,0,ODM[0,2]]]
  return np.array(M, dtype=np.float64)

#предыдущая функция, только наоборот
def MatToOneDim(MAT):
  M = [MAT[0][0],MAT[1][1],MAT[2][2]]
  return np.array(M, dtype=np.float64)