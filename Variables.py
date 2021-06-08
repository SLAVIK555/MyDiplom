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
import SimpleReturners as SRs

PREDICTOR_PATH = "/home/slava/Source/PredDiplom/shape_predictor_68_face_landmarks.dat"

if not os.path.isfile(PREDICTOR_PATH):
    print("[ERROR] USE models/downloader.sh to download the predictor")
    sys.exit()

face3Dmodel = SRs.ref3DModel()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

WorldCamReverse = 0
FaceCamReverse = 0
CamReverse = 0
Xreverse = 0
Zreverse = 0
XR = 0
NumOfFPv = 6

MC_image_points = np.array([
                            # (356, 266),#1
                            # (473, 276),#2
                            # (471, 372),#3
                            # (319, 352)#4
                            # (223, 293),#0
                            # (348, 307),#1
                            # (314, 394),#2
                            # (151, 373)#3
                            (342, 315),#0
                            (460, 331),#1
                            (446, 406),#2
                            (295, 386)#3
                        ], dtype="double")

SC_image_points = np.array([
                            # (361, 328),#0
                            # (230, 354),#1
                            # (158, 298),#2
                            # (276, 279)#3
                            # (486, 410),#0
                            # (314, 427),#1
                            # (286, 366),#2
                            # (424, 352)#3
                            (442, 364),#0
                            (302, 382),#1
                            (249, 315),#2
                            (367, 303)#3
                        ], dtype="double")

model_points = np.array([
##############################U####V####W####
                            # (0.0, 0.0, 0.0),#1
                            # (0.0, -17.4, 0.0),#2 1000/1300
                            # (-22.0, -17.4, 0.0),#3
                            # (-22.0, 0.0, 0.0)#4
                            (0.0, 0.0, 0.0),#1
                            (0.0, 17.4, 0.0),#2 1000/1300
                            (22.0, 17.4, 0.0),#3
                            (22.0, 0.0, 0.0)#4
                        ])
print ("MP:\n {0}".format(model_points[1][0]))

focal_length = 640
center = (640/2, 640/2)

default_camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

default_dist_coeffs = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)


MC_camera_matrix = np.array([[634.34310936639508, 0.0, 318.39139172190045],#?????????????????????????????????????????????????????????????????
                            [0.0, 634.63957523079807, 225.82312340854455],
                            [0.0, 0.0, 1.0]], dtype="double")
MC_dist_coeffs = np.array([[-1.8777317142602665e-01, 8.5462438684484254e-01, 2.3547270403845137e-04, 2.5241612322014512e-03, -1.0850762766772557e+00]], dtype=np.float64)

SC_camera_matrix = np.array([[853.44567517769258, 0.0, 358.30167030144056],#?????????????????????????????????????????????????????????????????
                            [0.0, 855.18699410234115, 282.35059869084620],
                            [0.0, 0.0, 1.0]], dtype="double")

SC_dist_coeffs = np.array([[1.0148856355600007e-01, 3.5830167030144056e-01, 1.3028397811836989e-02, 4.3977221471911167e-03, 4.0245221116215080e-01]], dtype=np.float64)