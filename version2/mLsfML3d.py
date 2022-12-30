# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

# import the necessary packages ------------------------------------------------

import argparse
import copy
import csv
from multiprocessing import Event
import cv2
import operator
import joblib
import math
import matplotlib
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import random as rd
import screeninfo as sc
import sklearn
import statistics
import sys
import threading
import time
import unidecode

from time import sleep
from time import *
import time

from PIL import ImageEnhance, Image
from threading import Thread

import mVideoConvert as vc

from text_to_speech import speak
from pathlib import Path
from sys import _getframe
from datetime import datetime

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler, QuantileTransformer, PowerTransformer, MaxAbsScaler
from sklearn.model_selection import train_test_split as data_split

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import RocCurveDisplay

from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe
from sklearn.feature_selection import chi2, f_classif, f_regression, r_regression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.neighbors import KNeighborsClassifier as KN
from sklearn.neighbors import NearestCentroid as NC
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier as DTC

from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import MiniBatchSparsePCA
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD

# ------------------------------------------------------------------------------
# Class for a normalizer identity
# ------------------------------------------------------------------------------
class cLsfNormalizerIdentity :

    def __init__(self, nClass=0) :
        self.nClass = nClass

    def fit_transform(self, aSamples) :
        return aSamples

    def transform(self, aSamples) :

        if self.nClass > 0 :
            nMax = max(aSample)
            nMin = min(aSample)
            return np.round((aSample - nMin) / (nMax - nMin) * self.nClass, 0)
        else :
            return aSamples

'''class Threads :
    def __init__(self, times, event):      # times = donnée supplémentaire
            threading.Thread.__init__(self)
            # (appel au constructeur de la classe mère)
            self.times = times           # donnée supplémentaire ajoutée à la classe
            self.state = False
            self.event = event

    def run(self):
        for i in range(0, self.jusqua):
            print("thread ", i)
            self.event.set()'''

# List of scalers to normalize data --------------------------------------------
aNormalizers_all = {
    "Identity"  : cLsfNormalizerIdentity(),
    "MaxAbs"    : MaxAbsScaler(), 
    "MinMax"    : MinMaxScaler(), 
    "Normalize" : Normalizer(), 
    "Power"     : PowerTransformer(), 
    "Quantile"  : QuantileTransformer(n_quantiles=100), 
    "Robust"    : RobustScaler(quantile_range=(25.0, 75.0)), 
    "Standard"  : StandardScaler()
}

aNormalizers_large = {
    "Identity"  : cLsfNormalizerIdentity(),
    "MaxAbs"    : MaxAbsScaler(), 
    "Power"     : PowerTransformer(), 
    "Quantile"  : QuantileTransformer(n_quantiles=100), 
    "Robust"    : RobustScaler(quantile_range=(25.0, 75.0)), 
    "Standard"  : StandardScaler()
}

aNormalizers_medium = {
    "Power"     : PowerTransformer(), 
    "Quantile"  : QuantileTransformer(n_quantiles=100), 
    "Standard"  : StandardScaler()
}

aNormalizers_small = {
    "Power"     : PowerTransformer(), 
    "Quantile"  : StandardScaler()
}

# List of classifiers ----------------------------------------------------------
aClassifiers_all = {
#    "ABC"        : ABC(),
    "DTC"        : DTC(max_depth=5),
    "GNB"        : GNB(),
#    "GPC"        : GPC(1.0 * RBF(1.0)),
    "KN_ball"    : KN(n_neighbors=5, weights='uniform', algorithm='ball_tree'),
    "LDA"        : LDA(),
    "MLP"        : MLP(alpha=1, max_iter=600, batch_size='auto'),
    "NC"         : NC(),
    "QDA"        : QDA(),
    "RFC"        : RFC(max_depth=5, n_estimators=10, max_features=1),
#    "SVM_linear" : SVM(kernel='linear', gamma='scale', C=1, cache_size=4000),
#    "SVM_poly"   : SVM(kernel='poly', degree=3, gamma='scale', C=1, cache_size=4000),
#    "SVM-RBF"    : SVM(kernel='rbf', gamma='scale', C=1, cache_size=4000)
}

aClassifiers_large = {
    "GNB"        : GNB(),
    "KN_ball"    : KN(n_neighbors=5, weights='uniform', algorithm='ball_tree'),
    "LDA"        : LDA(),
    "MLP"        : MLP(alpha=1, max_iter=600, batch_size='auto'),
}

aClassifiers_medium = {
    "KN_ball"    : KN(n_neighbors=5, weights='uniform', algorithm='ball_tree'),
    "LDA"        : LDA(),
    "MLP"        : MLP(alpha=1, max_iter=600, batch_size='auto'),
}

aClassifiers_small = {
    "KN_ball"    : KN(n_neighbors=5, weights='uniform', algorithm='ball_tree'),
    "MLP"        : MLP(alpha=1, max_iter=600, batch_size='auto'),
}

# Feature selection ------------------------------------------------------------
aSelectFeatures = {
    "fdr"        : SelectFdr,
    "fpr"        : SelectFpr,
    "fwe"        : SelectFwe,
    "kbest"      : SelectKBest,
    "percentile" : SelectPercentile,
}
aScoreFunctions = {
    "chi2"  : chi2,
    "fclas" : f_classif,
    #"freg"  : f_regression,
    #"rreg"  : r_regression,
}

# List of decomposition to reduct dimensions and visualize data ----------------
nComponents = 8
aDecompositions_all = {
    "FactorAnalysis"    : FactorAnalysis(n_components=nComponents), 
    "FastICA"           : FastICA(n_components=nComponents), 
    "IPCA"              : IncrementalPCA(n_components=nComponents),
    "KernelPCA"         : KernelPCA(n_components=nComponents),
    "LDA"               : LDA(n_components=nComponents),
    "LatentDirichlet"   : LatentDirichletAllocation(n_components=nComponents),
    "MiniBatchDict"     : MiniBatchDictionaryLearning(n_components=nComponents),
    "MiniBatchSpar"     : MiniBatchSparsePCA(n_components=nComponents),
    "NMF"               : NMF(n_components=nComponents),
    "PCA"               : PCA(n_components=nComponents), 
    "SparsePCA"         : SparsePCA(n_components=nComponents),
    "TruncatedSVD"      : TruncatedSVD(n_components=nComponents),
}

aDecompositions_large = {
    "PCA"               : PCA(n_components=nComponents), 
    "LDA"               : LDA(n_components=nComponents),
    "IPCA"              : IncrementalPCA(n_components=nComponents),
    "TruncatedSVD"      : TruncatedSVD(n_components=nComponents),
}

aDecompositions_medium = {
    "PCA"               : PCA(n_components=nComponents), 
    "LDA"               : LDA(n_components=nComponents),
    "TruncatedSVD"      : TruncatedSVD(n_components=nComponents),
}

aDecompositions_small = {
    "PCA"               : PCA(n_components=nComponents), 
    "LDA"               : LDA(n_components=nComponents),
}

# Global variables -------------------------------------------------------------

nWidth, nHeight = 800, 600

nWidthScreen = 1920
nHeightScreen = 1080
nWidthScreen = int(1920 / 1.25)
nHeightScreen = int(1080 / 1.25)

nSkipPourcentage = 0.05     # 15% or 5% of frame ignore at begining and end of record
nMostFreq = 5               # number of prédiction word to display

nConfidence = 0.50

# Landmark used in hands, pose, face mdels
aHandPoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#aPosePoints = [0, 11, 12, 13, 14, 15, 16, 19, 20]
#cP_NOSE, cP_LEFT_SHOULDER, cP_RIGHT_SHOULDER, cP_LEFT_ELBOW, cP_RIGHT_ELBOW, cP_LEFT_WRIST, cP_RIGHT_WRIST, cP_LEFT_INDEX, cP_Right_INDEX = 0, 1, 2, 3, 4, 5, 6, 7, 8
aPosePoints = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
cP_NOSE, cP_LEFT_SHOULDER, cP_RIGHT_SHOULDER, cP_LEFT_ELBOW, cP_RIGHT_ELBOW, cP_LEFT_WRIST, cP_RIGHT_WRIST, cP_LEFT_PINKY, cP_Right_PINKY, cP_LEFT_INDEX, cP_Right_INDEX, cP_LEFT_THUMB, cP_Right_THUMB = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
aFacePoints =  [19, 13, 14, 33, 263, 53, 283, 55, 285, 76, 306, 124, 353, 133, 362, 145, 374, 159, 386, 213, 433]
#cF_NOSE = 19
cF_NOSE = 0

# What group of landmarks we use : (0,1) = hands, (0, 1, 2) = hands + pose, (0, 1, 2, 3) = hands + pose + face
cLEFT_HAND = 0
cRIGHT_HAND = 1
cPOSE = 2
cFACE = 3

aLandmarkRef = [cP_LEFT_WRIST, cP_RIGHT_WRIST, 0, cP_NOSE]

bHandLeftVisiblePrev = False
bHandRightVisiblePrev = False

aHandLeftRest = [(-0.35119,0.59788,0.75578), (-0.09789,-0.08003,-0.0168), (-0.202,-0.11751,-0.0376), (-0.2979,-0.12989,-0.05454), (-0.36958,-0.14475,-0.07134), (-0.2684,-0.00507,-0.05705), (-0.39075,-0.00035,-0.07916), (-0.45537,0.00803,-0.08962), (-0.50294,0.01903,-0.09573), (-0.25873,0.07097,-0.06305), (-0.39999,0.09932,-0.07898), (-0.47415,0.11533,-0.0832), (-0.52323,0.12933,-0.08752), (-0.22983,0.13352,-0.06797), (-0.35723,0.1834,-0.08531), (-0.42859,0.2068,-0.08987), (-0.47994,0.22256,-0.09315), (-0.18831,0.17962,-0.07234), (-0.28691,0.23093,-0.08591), (-0.34651,0.25231,-0.08384), (-0.39619,0.26415,-0.08046)]
aHandRightRest = [(0.35119,0.59788,0.75578), (0.09789,-0.08003,-0.0168), (0.202,-0.11751,-0.0376), (0.2979,-0.12989,-0.05454), (0.36958,-0.14475,-0.07134), (0.2684,-0.00507,-0.05705), (0.39075,-0.00035,-0.07916), (0.45537,0.00803,-0.08962), (0.50294,0.01903,-0.09573), (0.25873,0.07097,-0.06305), (0.39999,0.09932,-0.07898), (0.47415,0.11533,-0.0832), (0.52323,0.12933,-0.08752), (0.22983,0.13352,-0.06797), (0.35723,0.1834,-0.08531), (0.42859,0.2068,-0.08987), (0.47994,0.22256,-0.09315), (0.18831,0.17962,-0.07234), (0.28691,0.23093,-0.08591), (0.34651,0.25231,-0.08384), (0.39619,0.26415,-0.08046)]

sWordDemo = ""
sWordDemoNew = ""
bNextVideo = True

aDisplResult = []
phrase = [" "]
sSentence = ""
nFont = cv2.FONT_HERSHEY_SIMPLEX
nFontScale = 1
nThickness = 2
nLineType = cv2.LINE_AA
cpt = 0

# ------------------------------------------------------------------------------
# Get screen size
# ------------------------------------------------------------------------------
def mLsfGetScreenSize() :
    #print(_getframe().f_code.co_name)

    # get the size of the screen
    for nIdxScreen in range(8) :
        try :
            oScreen = sc.get_monitors()[nIdxScreen]
            nWidthScreen, nHeightScreen = oScreen.width, oScreen.height    
            print("... nIdxScreen, nWidth, nHeight : ", nIdxScreen, nWidthScreen, nHeightScreen)
            break
        except :
            continue
    nWidthScreen = int(nWidthScreen / 1.25)
    nHeightScreen = int(nHeightScreen / 1.25)
    print("... nIdxScreen, nWidth, nHeight : ", nIdxScreen, nWidthScreen, nHeightScreen)

    return nWidthScreen, nHeightScreen

# ------------------------------------------------------------------------------
# Initialize camera
# ------------------------------------------------------------------------------
def mLsfInitCamera(nCamera = 0) :
    #print(_getframe().f_code.co_name)

    # Initialize the camera ----------------------------------------------------
    oCamera = cv2.VideoCapture(nCamera, cv2.CAP_DSHOW)

    # Choose resolution --------------------------------------------------------
    #nWidth, nHeight = 720, 480    # DVD
    #nWidth, nHeight = 1280, 720   # HD Ready
    #nWidth, nHeight = 1920, 1080  # Full HD
    #nWidth, nHeight = 3840, 2160  # 4K
    #nWidth, nHeight = 7680, 4320  # 8K

    nWidth, nHeight = 800, 600

    # Configure the height and the wight of the frames -------------------------
    oCamera.set(cv2.CAP_PROP_FRAME_WIDTH, nWidth)
    oCamera.set(cv2.CAP_PROP_FRAME_HEIGHT, nHeight)

    return oCamera, nWidth, nHeight

# ------------------------------------------------------------------------------
# mDiffImg() calculate distance between 2 images
# 1.0 images are identical
# 0.0 images are completely different
# ------------------------------------------------------------------------------
def mDiffImg(img1, img2, sMethod) :
    #print(_getframe().f_code.co_name)

    aMesDist = cv2.matchTemplate(img1, img2, sMethod)
    nMesDist = aMesDist[0][0]

    if (sMethod == cv2.TM_SQDIFF_NORMED) : nMesDist = 1 - nMesDist

    return nMesDist

# ------------------------------------------------------------------------------
# Functions to normalize features : landmark coordinates
# ------------------------------------------------------------------------------
def mPointCenter(aA, aB) :
    #print(_getframe().f_code.co_name)

    if (nDim == 2) : aCenter = ((aA[0] + aB[0])/2, (aA[1] + aB[1])/2)
    if (nDim == 3) : aCenter = ((aA[0] + aB[0])/2, (aA[1] + aB[1])/2, (aA[2] + aB[2])/2)
    return aCenter

def mPointDistance(aA, aB, nDimL = 3) :
    #print(_getframe().f_code.co_name)

    if (nDimL == 3) and (nDim == 3) :
        nDist = np.sqrt((aA[0] - aB[0])**2 + (aA[1] - aB[1])**2 + (aA[2] - aB[2])**2)
    else :
        nDist = np.sqrt((aA[0] - aB[0])**2 + (aA[1] - aB[1])**2)
    
    return nDist

def mPointNormalize(aPoints, aOrigin, nRefWx, nRefHy, nRefDz, nDilate = 1.0) :
    #print(_getframe().f_code.co_name)

    for nIdxPoint in range(len(aPoints)) :
        if (nDim == 2) : 
            aPoints[nIdxPoint] = (nDilate * (aPoints[nIdxPoint][0] - aOrigin[0]) / nRefWx, 
                                  nDilate * (aPoints[nIdxPoint][1] - aOrigin[1]) / nRefHy)
        if (nDim == 3) : 
            aPoints[nIdxPoint] = (nDilate * (aPoints[nIdxPoint][0] - aOrigin[0]) / nRefWx, 
                                  nDilate * (aPoints[nIdxPoint][1] - aOrigin[1]) / nRefHy,
                                  nDilate * (aPoints[nIdxPoint][2] - aOrigin[2]) / nRefDz)
    return aPoints

# ------------------------------------------------------------------------------
# Initilize mediapipe models : hands, pose, face
# ------------------------------------------------------------------------------
def mLsfInitMediaPipe(sAction) :
    #print(_getframe().f_code.co_name)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    mp_holistic = mp.solutions.holistic

    #holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=2, refine_face_landmarks=True, min_detection_confidence=nConfidence, min_tracking_confidence=nConfidence)
    if sAction == "extr" : 
        holistic = mp_holistic.Holistic(static_image_mode=True, model_complexity=1, refine_face_landmarks=True, min_detection_confidence=nConfidence, min_tracking_confidence=nConfidence)
    else :
        holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1, refine_face_landmarks=True, min_detection_confidence=nConfidence, min_tracking_confidence=nConfidence)

    return mp_drawing, mp_drawing_styles, mp_holistic, holistic

# ------------------------------------------------------------------------------
# Interpolate coordinates between 2 known positions of hands or face
# ------------------------------------------------------------------------------
def mInterpolatePart(cLM, nIdxVector, nLast, nLong, nIdxLoop, aSequences) :       
    #print(_getframe().f_code.co_name)

    nEmpty = 0
    for nIdxMissing in range(nLast + 1, nIdxLoop) :

        # Proportionality coefficients
        if (sRefXpolate == "linear") or (nIdxVector == cFACE):
            # Linear Interpolation : Linear shift at this point
            # Mandatory for face interpolation (nIdxVector == cFACE)
            n0Proportion = (nIdxMissing - nLast) / (nIdxLoop - nLast)
            n1Proportion = n0Proportion
            n2Proportion = n0Proportion
        else :
            # Interpolation using known wrist position : Proportional wrist shift at this point
            # [cPOSE] = groupe pose dont on prend le poignet [cLM]
            try : n0Proportion = (aSequences[nIdxMissing][cPOSE][cLM][0] - aSequences[nLast][cPOSE][cLM][0]) / (aSequences[nIdxLoop][cPOSE][cLM][0] - aSequences[nLast][cPOSE][cLM][0])
            except : n0Proportion = (nIdxMissing - nLast) / (nIdxLoop - nLast)
            try : n1Proportion = (aSequences[nIdxMissing][cPOSE][cLM][1] - aSequences[nLast][cPOSE][cLM][1]) / (aSequences[nIdxLoop][cPOSE][cLM][1] - aSequences[nLast][cPOSE][cLM][1])
            except : n1Proportion = (nIdxMissing - nLast) / (nIdxLoop - nLast)
            if (nDim == 3) :
                try : n2Proportion = (aSequences[nIdxMissing][cPOSE][cLM][2] - aSequences[nLast][cPOSE][cLM][2]) / (aSequences[nIdxLoop][cPOSE][cLM][2] - aSequences[nLast][cPOSE][cLM][2])
                except : n2Proportion = (nIdxMissing - nLast) / (nIdxLoop - nLast)

        # Interpolate the feature vector
        aSequence = []
        for nIdxLm in range(nLong) :
            if (nDim == 2) :
                aSequence.append((aSequences[nLast][nIdxVector][nIdxLm][0] + n0Proportion * (aSequences[nIdxLoop][nIdxVector][nIdxLm][0] - aSequences[nLast][nIdxVector][nIdxLm][0]),
                                  aSequences[nLast][nIdxVector][nIdxLm][1] + n1Proportion * (aSequences[nIdxLoop][nIdxVector][nIdxLm][1] - aSequences[nLast][nIdxVector][nIdxLm][1])))
            if (nDim == 3) :
                aSequence.append((aSequences[nLast][nIdxVector][nIdxLm][0] + n0Proportion * (aSequences[nIdxLoop][nIdxVector][nIdxLm][0] - aSequences[nLast][nIdxVector][nIdxLm][0]),
                                  aSequences[nLast][nIdxVector][nIdxLm][1] + n1Proportion * (aSequences[nIdxLoop][nIdxVector][nIdxLm][1] - aSequences[nLast][nIdxVector][nIdxLm][1]),
                                  aSequences[nLast][nIdxVector][nIdxLm][2] + n2Proportion * (aSequences[nIdxLoop][nIdxVector][nIdxLm][2] - aSequences[nLast][nIdxVector][nIdxLm][2])))

        aSequences[nIdxMissing][nIdxVector].append(np.asarray(aSequence).reshape(nLong, nDim))
        nEmpty += 1

    return nEmpty

# ------------------------------------------------------------------------------
# Distance between 2 feature vectors
# If only 1 vector filled get à static value for distance
# ------------------------------------------------------------------------------
def mGetDistance(aFeatureP, aFeatureC) :       
    #print(_getframe().f_code.co_name)

    aDistance = []
    for nIdxVector in aVectors :
        #print("... ", len(aFeatureP[nIdxVector]), len(aFeatureC[nIdxVector]))
        nShape = aShapes[nIdxVector]
        if (len(aFeatureP[nIdxVector]) > 0) and (len(aFeatureC[nIdxVector]) > 0) :
            aPrevious = np.asarray(aFeatureP[nIdxVector]).reshape(nShape)
            aCurrent = np.asarray(aFeatureC[nIdxVector]).reshape(nShape)
            nDistance = np.linalg.norm(aPrevious - aCurrent, ord=nDim, axis=0.)
            aDistance.append(nDistance)
        elif (len(aFeatureP[nIdxVector]) > 0) or (len(aFeatureC[nIdxVector]) > 0) :
            if (nIdxVector == cLEFT_HAND) or (nIdxVector == cRIGHT_HAND) : aDistance.append(2*aDistThreshold[nIdxVector])
            if (nIdxVector == cPOSE) : aDistance.append(2*aDistThreshold[nIdxVector])
            if (nIdxVector == cFACE) : aDistance.append(2*aDistThreshold[nIdxVector])
        else :
            aDistance.append(0)
    return aDistance

# ------------------------------------------------------------------------------
# Extract features (hands, pose and face) from image
# ------------------------------------------------------------------------------
def mGetFeatures(image, sWord) :
    #print(_getframe().f_code.co_name)
        
    global bHandLeftVisiblePrev, bHandRightVisiblePrev, aDisplResult, phrase, sSentence, cpt

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try :
        holisticResults = holistic.process(image)
    except :
        return False, None, None, None, None, False, False, False, False

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get and draw the pose annotation on the image ----------------------------
    aPoses = []
    if holisticResults.pose_landmarks :
        if ("pose" in aDisplay) :
            mp_drawing.draw_landmarks(
                image,
                holisticResults.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Keep 9/33 upper body pose landmarks --------------------------------------
        try :
            for nIdxPose, lm in enumerate(holisticResults.pose_landmarks.landmark) :

                if (nIdxPose in aPosePoints) :
                    if (nDim == 2) : aPoses.append((lm.x, lm.y))
                    if (nDim == 3) : aPoses.append((lm.x, lm.y, lm.z))
                    """
                    # x = h height, y = w width
                    if (nIdxPose in (0, 11, 12,)) : 
                        print(nIdxPose, lm.x, lm.y)
                        print()
                    """

        except :
            return False, None, None, None, None, False, False, False, False

        # Calculate appear and disappear hands detection for pose --------------
        bHandLeftAppear, bHandLeftDisAppear = False, False
        nHandLeftVisible = holisticResults.pose_landmarks.landmark[15].visibility
        if (nHandLeftVisible > 0.55) and not bHandLeftVisiblePrev :
            bHandLeftAppear = True
            bHandLeftVisiblePrev = True
        elif (nHandLeftVisible < 0.45) and bHandLeftVisiblePrev :
            bHandLeftDisAppear = True
            bHandLeftVisiblePrev = False
        bHandLeftVisible = (nHandLeftVisible > nAttemptThreshold)

        bHandRightAppear, bHandRightDisAppear = False, False
        nHandRightVisible = holisticResults.pose_landmarks.landmark[16].visibility
        if (nHandRightVisible > 0.55) and not bHandRightVisiblePrev :
            bHandRightAppear = True
            bHandRightVisiblePrev = True
        elif (nHandRightVisible < 0.45) and bHandRightVisiblePrev :
            bHandRightDisAppear = True
            bHandRightVisiblePrev = False
        bHandRightVisible = (nHandRightVisible > nAttemptThreshold)

        # Attempt X time, when no hands detected -----------------------------------
        aHandLeft = []
        nLoopLeft = 1
        while (True) :

            # Get and draw the hands annotations on the image ----------------------
            aHandLeft = []
            if holisticResults.left_hand_landmarks :
                if ("hands" in aDisplay) :
                    mp_drawing.draw_landmarks(
                        image,
                        holisticResults.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())

                # Keep all landmarks -----------------------------------------------
                try :
                    for lm in holisticResults.left_hand_landmarks.landmark :

                        if (nDim == 2) : aHandLeft.append((lm.x, lm.y))
                        if (nDim == 3) : aHandLeft.append((lm.x, lm.y, lm.z))

                except :
                    return False, None, None, None, None, False, False, False, False

            if (len(aHandLeft) > 0) or not bHandLeftVisible : break
            nLoopLeft += 1
            if (nLoopLeft > nAttemptFeat) : break     # nAttemptFeat try to get features

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try :
                holisticResults = holistic.process(image)
            except :
                return False, None, None, None, None, False, False, False, False

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Attempt X time, when no hands detected -----------------------------------
        aHandRight = []
        nLoopRight = 1
        while (True) :

            aHandRight = []
            if holisticResults.right_hand_landmarks :
                if ("hands" in aDisplay) :
                    mp_drawing.draw_landmarks(
                        image,
                        holisticResults.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())

                # Keep all landmarks -----------------------------------------------
                try :
                    for lm in holisticResults.right_hand_landmarks.landmark :

                        if (nDim == 2) : aHandRight.append((lm.x, lm.y))
                        if (nDim == 3) : aHandRight.append((lm.x, lm.y, lm.z))

                except :
                    return False, None, None, None, None, False, False, False, False

            if (len(aHandRight) > 0) or not bHandRightVisible : break
            nLoopRight += 1
            if (nLoopRight > nAttemptFeat) : break     # nAttemptFeat try to get features

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try :
                holisticResults = holistic.process(image)
            except :
                return False, None, None, None, None, False, False, False, False

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Get and draw the face annotations on the image ---------------------------
        aFaces = []
        if holisticResults.face_landmarks :
            if ("face" in aDisplay) :
                mp_drawing.draw_landmarks(
                    image,
                    holisticResults.face_landmarks,
                    mp_holistic.FACE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_face_landmarks_style())

            # Keep only pertinent landmarks ------------------------------------
            try :
                for nIdxFace, lm in enumerate(holisticResults.face_landmarks.landmark) :

                    if (nIdxFace in aFacePoints) :
                        if (nDim == 2) : aFaces.append((lm.x, lm.y))
                        if (nDim == 3) : aFaces.append((lm.x, lm.y, lm.z))

            except :
                return False, None, None, None, None, False, False, False, False
            
    else :
        return False, None, None, None, None, False, False, False, False

    #if ("hands" in aDisplay) or ("pose" in aDisplay) or ("face" in aDisplay) :
    
    cv2.namedWindow(sWord)
    cv2.moveWindow(sWord, nWidthScreen//2, 0)    
    nIdxLine = 1
    #for sWordL, number in aDisplResult:
    """ if (sWordL != phrase[-1]) :
        if not(bHandRightDisAppear and bHandLeftDisAppear) :
            cpt += 1
            phrase.append(sWordL)
            sSentence = ""
        if (bHandRightDisAppear or bHandLeftDisAppear) or (bHandRightDisAppear and bHandLeftDisAppear) or (cpt > 4):
            phrase.clear()
            phrase = [" "]
            cpt = 0"""
    for sWordL, number in aDisplResult:
        if (sWordL != phrase[-1]) :
            if not(bHandRightDisAppear and bHandLeftDisAppear) :
                cpt += 1
                phrase.append(sWordL)
                sSentence = ""
            """if (bHandRightDisAppear or bHandLeftDisAppear) or (bHandRightDisAppear and bHandLeftDisAppear) or (cpt > 4):
                phrase.clear()
                phrase = [" "]
                cpt = 0"""      
        sSentence = ""
        for e in phrase :
            sSentence += e + ' '
        if (len(sSentence) != 0) :
            nb_carac = len(sSentence)
        if (nb_carac > 42 ) :
            phrase.pop(1)
        cv2.putText(image, f"{(unidecode.unidecode(sSentence))}", (0, 45*nIdxLine), nFont, nFontScale, (0, 0, 255), nThickness, nLineType)
        nIdxLine += 1
        #if nIdxLine > 2 : break
        if nIdxLine > 3 : break
        

    cv2.imshow(sWord, image)
    #if cv2.waitKey(nWaitTime) == ord('q'): pass
    if cv2.waitKey(nWaitTime) == ord('q'): exit()

    #print("... delay : ", time.time() - nStartTime)

    """
    print("... aHandLeft : ", len(aHandLeft))
    print("... aHandRight : ", len(aHandRight))
    print("... aPoses : ", len(aPoses))
    print("... aFaces : ", len(aFaces))
    print("... aHandLeft, aHandRight : ", len(aHandLeft), len(aHandRight))
    """
    
    if len(aHandLeft) > 0 : bHandLeftVisiblePrev = True
    if len(aHandRight) > 0 : bHandRightVisiblePrev = True

    return True, aHandLeft, aHandRight, aPoses, aFaces, bHandLeftAppear, bHandLeftDisAppear, bHandRightAppear, bHandRightDisAppear

# ------------------------------------------------------------------------------
# Normalize features try to get features independent from distance and people
# ------------------------------------------------------------------------------
def mNormalizeFeatures(aHandLeft, aHandRight, aPoses, aFaces) :
    #print(_getframe().f_code.co_name)

    # Normalisation : origin middle of shoulders -------------------------------
    # width  scale: nRefWx = shoulder to shoulder
    # height scale: nRefHy = nose to middle of shoulders
    aOrigin = mPointCenter(aPoses[cP_LEFT_SHOULDER], aPoses[cP_RIGHT_SHOULDER])
    nRefWx = mPointDistance(aPoses[cP_LEFT_SHOULDER], aPoses[cP_RIGHT_SHOULDER])
    nRefHy = mPointDistance(aPoses[cP_NOSE], aOrigin)

    # Use relative sizes -------------------------------------------------------
    aPoses = mPointNormalize(aPoses, aOrigin, nRefWx, nRefHy, nRefWx)
    #print("... nose, left shoulder, right shoulder, aOrigin, nRefWx, nRefHy : ", aPoses[0], aPoses[1], aPoses[2], aOrigin, nRefWx, nRefHy)

    aFaces = mPointNormalize(aFaces, aOrigin, nRefWx, nRefHy, nRefWx)

    # For hands, use wrist as origin after normalisation vs pose
    if (len(aHandLeft) > 0) :
        aHandLeft = mPointNormalize(aHandLeft, aOrigin, nRefWx, nRefHy, nRefWx)
        aHandLeft[1:] = mPointNormalize(aHandLeft[1:], aHandLeft[0], 1.0, 1.0, 1.0, 1.0)
        #print("... aHandLeftRest : ", aHandLeft)

    if (len(aHandRight) > 0) :
        aHandRight = mPointNormalize(aHandRight, aOrigin, nRefWx, nRefHy, nRefWx)
        aHandRight[1:] = mPointNormalize(aHandRight[1:], aHandRight[0], 1.0, 1.0, 1.0, 1.0)
        #print("... aHandRightRest : ", aHandRight)

    return aHandLeft, aHandRight, aPoses, aFaces

# ------------------------------------------------------------------------------
# Keep frames when its distance from previous one is > threshold
# Select or expand to keep just nSample samples
# ------------------------------------------------------------------------------
def mLsfFilterFrames(sFile, aSequences, nSequencesInit) :
    #print(_getframe().f_code.co_name)

    aFeatures = []
    nSequences = len(aSequences)
    bDistThreshold = False

    # Keep first and last landmark, and when hands or pose or face move > threshold
    aFeatures.append(aSequences[0])
    for nIdxFrame in range(1, nSequences) :
        aDistance = mGetDistance(aFeatures[-1], aSequences[nIdxFrame])

        # Keep this frame if at least one distance is greater than threshold
        # Only the distance of the hands is important
        bDistThreshold = False
        # bde test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #for nIdxVector in aVectors[0:2] :  # hands
        #for nIdxVector in aVectors[2:3] :  # pose
        for nIdxVector in aVectors[0:3] :  # hands + pose
            if aDistance[nIdxVector] > aDistThreshold[nIdxVector] :
                bDistThreshold = True
                break
        if bDistThreshold :
            aFeatures.append(aSequences[nIdxFrame])
    if not bDistThreshold : aFeatures.append(aSequences[-1])

    nRowFeatures = len(aFeatures)

    # In case of word recognition, skip first and last X% trames (10 to 25%)
    nSkip = int(round(nSkipPourcentage * nRowFeatures))
    aFeatures = aFeatures[nSkip : nRowFeatures-nSkip]
    nFeatures = len(aFeatures)

    # Expand / dilate data when not enough data
    if (nFeatures < nSample) : 
        aSmaller = copy.deepcopy(aFeatures)
        aFeatures = []
        for nIdxFrame in range(nSample) :
            nIdx = int(nIdxFrame / nSample * nFeatures)
            #print(nIdx, nIdxFrame, nSample, nFeatures)
            aFeatures.append(aSmaller[nIdx])
    nFeatures = len(aFeatures)

    # For test case, we select nSample frames for a word, static value firstly
    # we sample these "nSample" from all frames recorded
    aSamples = []
    nStep = (nFeatures - 1) // (nSample - 1)
    nOffSet = (nFeatures - nSample - (nStep - 1) * (nSample - 1)) // 2

    # Serialize nSample into a unique vector
    nLoop = 1
    for nIdxFrame in range(nOffSet, nFeatures, nStep) :
        for nIdxVector in aVectors :
            nShape = aShapes[nIdxVector]
            if (len(aFeatures[nIdxFrame][nIdxVector]) > 0) :
                aOne = np.asarray(aFeatures[nIdxFrame][nIdxVector]).reshape(nShape).tolist()
            else :
                aOne = np.zeros(nShape).tolist()
            aSamples = aSamples + aOne
        nLoop += 1
        if (nLoop > nSample) : break

    # bde test - for data cleanup <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    if (sAction in ("feat", "all*")) :
        print("... sFile, nSequencesInit, nSequences, nRowFeatures, nFeatures, nSample, nOffSet, nStep, len() : ", sFile, nSequencesInit, nSequences, nRowFeatures, nFeatures, nSample, nOffSet, nStep, len(aSamples))

    return aSamples, nSequences, nRowFeatures, nFeatures
    
# ------------------------------------------------------------------------------
# Equalizer histogram
# ------------------------------------------------------------------------------
def mEqualizeHist(imgIn) :

    imgYUV = cv2.cvtColor(imgIn, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    imgYUV[:,:,0] = cv2.equalizeHist(imgYUV[:,:,0])

    # convert the YUV image back to RGB format
    imgOut = cv2.cvtColor(imgYUV, cv2.COLOR_YUV2BGR)

    return imgOut
    
# ------------------------------------------------------------------------------
# Exract raw feature
# ------------------------------------------------------------------------------
kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
aPrattKernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])

def mLsfRawFeature(sDirIn, sSrc, sDirModel, sDataFrame, mp_holistic, holistic) :
    #print(_getframe().f_code.co_name)

    nStartTime = time.time()
    aFileSequences = []

    for sSrc in aSrc :
        sDirSrc = sDirIn + sSrc + "/"

        aFiles = []
        for sFile in os.listdir(sDirSrc):
            if (not os.path.isfile(sDirSrc + sFile)) : continue
            aFiles.append(sFile)

        nStartTime = time.time()

        nIdxFile = -1
        nFileMax = len(aFiles)
        nFile = 0

        print("... len() {} / {} ".format(nFileLimit, nFileMax))

        while (True) :

            # Rule to select input file ----------------------------------------
            if (sPlayList == "alea") : 
                #nIdxFile = int(nFileMax * rd.random())
                nIdxFile = rd.randint(0, nFileMax)
            else :
                nIdxFile += 1
                if (nIdxFile >= nFileMax) : break

            sFile = aFiles[nIdxFile]
            sFileExt = sFile[-4:]
            if (sFileExt != ".mp4") and (sFileExt != ".jpg") : 
                continue

            nFile += 1

            sBaseFile = os.path.basename(sFile)
            sWord = sBaseFile.split("#")[0]
            #print("... sBaseFile : ", sBaseFile, sWord, nIdxFile)

            for sMode in ('original', 'flip') :

                # Open stream on the file --------------------------------------
                if (sFileExt == ".mp4") :
                    camera = cv2.VideoCapture(sDirSrc + sFile)
                    sCR = camera.set(cv2.CAP_PROP_FPS, nFPS)
                    #print("... FPS : ", camera.get(cv2.CAP_PROP_FPS))

                aSequences = []
                nHandEmpty = 0
                aLastFrame = [-1, -1, -1, -1]

                # Analyze each frame -------------------------------------------
                nFileIn = 0
                while True :

                    if (sFileExt == ".mp4") :
                        success, image = camera.read()
                        if not success : break

                        sCR = camera.set(cv2.CAP_PROP_FPS, nFPS)
                        
                    else :
                        if nFileIn > nSample : break
                        nFileIn += 1

                        image = cv2.imread(sDirSrc + sFile)

                    image = cv2.resize(image, (nWidth, nHeight))
                    if (sMode == 'flip') and (sWord not in ("gauche", "droite", "question")) : image = cv2.flip(image, 1)

                    """
                    # Try to enhance image -------------------------------------
                    image = cv2.blur(image, (5,5))
                    image = cv2.GaussianBlur(image, (5,5), 0)
                    image = cv2.medianBlur(image, 5)
                    image = cv2.filter2D(image, -1, aPrattKernel)
                    image = mEqualizeHist(image)

                    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
                    image = np.array(ImageEnhance.Sharpness(image_pil).enhance(2))

                    cv2.imshow("bde", image)
                    cv2.waitKey(0)
                    """

                    # Get features from image ----------------------------------
                    bCR, aHandLeft, aHandRight, aPoses, aFaces, bHandLeftAppear, bHandLeftDisAppear, bHandRightAppear, bHandRightDisAppear = mGetFeatures(image, sWord)
                    if not bCR : 
                        break

                    # Aggregate pose, and hands features into a unique structure
                    # index 0: left hand, index 1: right hand, index 2: pose, index 3: face, 
                    aSequences.append((aHandLeft, aHandRight, aPoses, aFaces, (bHandLeftAppear, bHandLeftDisAppear, bHandRightAppear, bHandRightDisAppear)))

                # Clean everything before next loop ----------------------------
                if (sFileExt == ".mp4") :
                    camera.release()
                cv2.destroyAllWindows()

                # One video...
                aFileSequences.append((sFile, aSequences))

                print("... sBaseFile, sWord, len(aSequences) : ", sBaseFile, sWord, len(aSequences))
            
            if nFile > nFileLimit : break

    # Save raw feature extraction ----------------------------------------------
    sDataFile = sDirModel + sSrc + ".npy"
    joblib.dump(aFileSequences, sDataFile)

    nDelay = time.time() - nStartTime
    print("... delay : ", nDelay, nDelay / nFile)
    
# ------------------------------------------------------------------------------
# Split extracted features into differents dataset
# ------------------------------------------------------------------------------
def mLsfSplit(sDirIn, sSrc, sDirModel) :
    #print(_getframe().f_code.co_name)

    nStartTime = time.time()

    # Load raw feature extraction ----------------------------------------------
    sDataFile = sDirModel + "lsf-0260.npy"
    aFileSequencesRef = np.load(sDataFile, allow_pickle=True)

    for sDir in os.listdir("./select/"):

        sRoot = sDir[0:4]
        sNum = sDir[4:] if sDir[4:].isnumeric() else "0"
        nNum = int(sNum)

        if (sRoot != "lsf-") or (nNum == 0) or (nNum == 260) : continue
        print("... Split for directory : ", sDir)

        aFileSequences = []
        nFile = 0
        for sFileRef in os.listdir("./select/" + sDir):
            #print("... sFileRef : ", sFileRef)
            
            if sDir in ("lsf-005") :
                # bde test - like internet extraction ------------------------------
                for sFile, aSequence in aFileSequencesRef :
                    if (sFile == sFileRef) :
                        # Without flip
                        if ((nFile % 8) in (0, )) : aFileSequences.append((sFile, aSequence))
                        # With flip
                        #if ((nFile % 8) in (0, 1)) : aFileSequences.append((sFile, aSequence))

                        nFile += 1
            else :
                for sFile, aSequence in aFileSequencesRef :
                    if (sFile == sFileRef) :
                        aFileSequences.append((sFile, aSequence))

        # Save raw feature extraction ------------------------------------------
        sDataFile = sDirModel + sDir + ".npy"
        print("... sDataFile : ", sDataFile, len(aFileSequences))
        joblib.dump(aFileSequences, sDataFile)

    nDelay = time.time() - nStartTime
    print("... delay : ", nDelay)

# ------------------------------------------------------------------------------
# Normalize and complet features
# ------------------------------------------------------------------------------
def mLsfFeature(sDirIn, sSrc, sDirModel, sDataFrame, mp_holistic, holistic) :
    #print(_getframe().f_code.co_name)

    nStartTime = time.time()
    aLstFeatures = []

    # Load raw feature extraction ----------------------------------------------
    sDataFile = sDirModel + sSrc + ".npy"
    aFileSequences = np.load(sDataFile, allow_pickle=True)

    # For each video -----------------------------------------------------------
    for aFileSequence in aFileSequences :

        """
        # Select only 5% of files
        if (sPlayList == "alea") : 
            if (rd.random() > 0.05) : continue
        """

        sFile = aFileSequence[0]
        aVideoSequence = aFileSequence[1]
        
        sBaseFile = os.path.basename(sFile)
        sWord = sBaseFile.split("#")[0]
        #print("... sFile : ", sFile, sWord)

        aSequences = []
        nHandEmpty = 0
        aLastFrame = [-1, -1, -1, -1]

        nIdxLoop = 0
        for aHandLeft, aHandRight, aPoses, aFaces, (bHandLeftAppear, bHandLeftDisAppear, bHandRightAppear, bHandRightDisAppear) in aVideoSequence :

            # Normalize feature : origin middle of shoulders, etc.  ------------
            aHandLeft, aHandRight, aPoses, aFaces = mNormalizeFeatures(aHandLeft, aHandRight, aPoses, aFaces)

            # Extrapolate missing features -------------------------------------
            # Add default hand when no hand.landmark but pose.wrist
            if bExtrapolateHands :
                if (len(aHandLeft) == 0) and (bHandLeftAppear or bHandLeftDisAppear) :
                    aHandLeft = aHandLeftRest
                if (len(aHandRight) == 0) and (bHandRightAppear or bHandRightDisAppear) :
                    aHandRight = aHandRightRest

            # Aggregate pose, and hands features into a unique structure
            # index 0: left hand, index 1: right hand, index 2: pose, index 3: face, 
            aSequences.append((aHandLeft, aHandRight, aPoses, aFaces))

            # Interpolate missing features -------------------------------------
            # No interpolation for pose and face landmarks
            if bInterpolateHands :
                for nIdxVector in (cLEFT_HAND, cRIGHT_HAND) :

                    nLength = len(aSequences[-1][nIdxVector])
                    if nLength > 0 :
                        nLast = aLastFrame[nIdxVector]
                        if nLast < 0 :
                            nLast = nIdxLoop

                        elif nLast < (nIdxLoop - 1) :
                            # Interpolation needed
                            nHandEmpty += mInterpolatePart(aLandmarkRef[nIdxVector], nIdxVector, nLast, nLength, nIdxLoop, aSequences)

                        aLastFrame[nIdxVector] = nIdxLoop

            nIdxLoop += 1

        # Suppress sequence features without hands -----------------------------
        aSequencesCopy = []
        nSequencesInit = len(aSequences)
        for aSequence in aSequences :
            if (len(aSequence[cLEFT_HAND]) != 0) or (len(aSequence[cRIGHT_HAND]) != 0) :
                aSequencesCopy.append(aSequence)
        aSequences = aSequencesCopy

        #print("... len(aSequences) : ", len(aSequences))
        nSequences = len(aSequences)

        # Before filtering and calculation, check that we have data to do that...
        if (nSequences < 1) :
            # After the feature extraction we having no data => it's an error !
            # bde test - for data cleanup <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            if (sAction in ("feat", "all*")) :
                print("... sFile, nSequencesInit, nSequences, nRowFeatures, nFeatures, nSample, nOffSet, nStep, len() : ", sFile, nSequencesInit, nSequences, 0, 0, nSample, 0, 0, 0)

        else :

            # Keep landmarks distance of minimal threshold...
            aSamples, nSequences, nRowFeatures, nFeatures = mLsfFilterFrames(sFile, aSequences, nSequencesInit)

            aLstFeatures.append([sWord] + aSamples)

            # ------------------------------------------------------------------
            # Feature augmentation : different scales for hands or pose, but the face doesn't change
            # ------------------------------------------------------------------
            def mDataAugmented(aSamplesRef, nIeme, sDataAugmentation) :
                #print(_getframe().f_code.co_name)

                aSamples = copy.deepcopy(aSamplesRef)
                
                # Frame length
                n1Sample = aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND] + aShapes[cPOSE] + aShapes[cFACE]
                n1Sample = sum(aShapes)
                #print("... n1Sample : ", n1Sample)

                # Hands pose and face shift : +/- 7 or 14% ---------------------
                if ("shift" == sDataAugmentation) :
                    nCoeffX = (2 * rd.random() - 1) * 0.25/3.35
                    nCoeffY = (2 * rd.random() - 1) * 0.50/3.35

                    # Hands shift ----------------------------------------------
                    # Only wrists move (0 : left, aShapes[cLEFT_HAND] : right) x N=8 frames
                    for nIdxSample in range(nSample) :
                        nStart = nIdxSample * n1Sample
                        nEnd = nIdxSample * n1Sample + (aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND])
                        for nIdx in range(nStart, nEnd, aShapes[cLEFT_HAND]) :
                            nDeltaX = nCoeffX * (aSamplesRef[nIdx+1]+1)
                            if abs(aSamplesRef[nIdx]) > 0.001 :
                                aSamples[nIdx] = aSamplesRef[nIdx] + nDeltaX
                            nDeltaY = nCoeffY * (aSamplesRef[nIdx+1]+1)
                            if abs(aSamplesRef[nIdx+1]) > 0.001 :
                                aSamples[nIdx+1] = aSamplesRef[nIdx+1] + nDeltaY
                            #print(f'... shift hand {nIdx} : {nDeltaX:5.3f} - {aSamplesRef[nIdx]:5.3f} => {aSamples[nIdx]:5.3f}; {nDeltaY:5.3f} / {aSamplesRef[nIdx+1]:5.3f} => {aSamples[nIdx+1]:5.3f}')
                    #print()

                    # Pose shift -----------------------------------------------
                    # Only arm and wrists move (no nose and shoulders = 3 points * 3 axes)
                    if (len(aVectors) > cPOSE) :
                        for nIdxSample in range(nSample) :
                            nStart = nIdxSample * n1Sample + aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND] + 3*nDim
                            nEnd = nIdxSample * n1Sample + aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND] + aShapes[cPOSE]
                            # For all values
                            for nIdx in range(nStart, nEnd, nDim) :
                                nDeltaX = nCoeffX * (aSamplesRef[nIdx+1]+1)
                                if abs(aSamplesRef[nIdx]) > 0.001 :
                                    aSamples[nIdx] = aSamplesRef[nIdx] + nDeltaX
                                nDeltaY = nCoeffY * (aSamplesRef[nIdx+1]+1)
                                if abs(aSamplesRef[nIdx+1]) > 0.001 :
                                    aSamples[nIdx+1] = aSamplesRef[nIdx+1] + nDeltaY
                                #print(f'... shift pose {nIdx} : {nDeltaX:5.3f} - {aSamplesRef[nIdx]:5.3f} => {aSamples[nIdx]:5.3f}; {nDeltaY:5.3f} / {aSamplesRef[nIdx+1]:5.3f} => {aSamples[nIdx+1]:5.3f}')
                            #print()

                # Hands pose and face noise : +/- 1% ---------------------------
                if ("noise" == sDataAugmentation) :
                    nNoiseVar = 0.011
                    
                    # Hands ----------------------------------------------------
                    nNoise = 1.0 + (2 * rd.random() - 1) * nNoiseVar
                    for nIdxSample in range(nSample) :
                        nStart = nIdxSample * n1Sample
                        nEnd = nIdxSample * n1Sample + (aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND])
                        for nIdx in range(nStart, nEnd) :
                            aSamples[nIdx] = nNoise * aSamplesRef[nIdx]
                            #print(f'... noise hand {nIdx} : {nNoise:5.3f} - {aSamplesRef[nIdx]:5.3f} => {aSamples[nIdx]:5.3f}')
                        #print()
                    
                    # Pose -----------------------------------------------------
                    if (len(aVectors) > cPOSE) : # check if pose (idx 2) is taking in count
                    
                        nNoise = 1.0 + (2 * rd.random() - 1) * nNoiseVar
                        for nIdxSample in range(nSample) :
                            nStart = nIdxSample * n1Sample + aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND]
                            nEnd = nIdxSample * n1Sample + aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND] + aShapes[cPOSE]
                            for nIdx in range(nStart, nEnd) :
                                aSamples[nIdx] = nNoise * aSamplesRef[nIdx]
                                #print(f'... noise pose {nIdx} : {nNoise:5.3f} - {aSamplesRef[nIdx]:5.3f} => {aSamples[nIdx]:5.3f}')
                            #print()

                    """
                    # Face -----------------------------------------------------
                    if (len(aVectors) > cFACE) : # check (idx cFACE) if face is taking in count
                    
                        nNoise = 1.0 + (2 * rd.random() - 1) * nNoiseVar
                        for nIdxSample in range(nSample) :
                            nStart = nIdxSample * n1Sample + aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND] + aShapes[cPOSE]
                            nEnd = nIdxSample * n1Sample + aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND] + aShapes[cPOSE] + aShapes[cFACE]
                            for nIdx in range(nStart, nEnd) :
                                aSamples[nIdx] = nNoise * aSamplesRef[nIdx]
                                #print(f'... noise face {nIdx} : {nNoise:5.3f} - {aSamplesRef[nIdx]:5.3f} => {aSamples[nIdx]:5.3f}')
                            #print()
                    """

                # Hands, pose and face scale -----------------------------------
                if ("scale" == sDataAugmentation) :
                    # Hands scale : +/- 15% ------------------------------------
                    nHandSizeVar = 0.14
                    nHandScale = 1.0 + (2 * rd.random() - 1) * nHandSizeVar

                    for nIdxSample in range(nSample) :
                        nStart = nIdxSample * n1Sample
                        nEnd = nIdxSample * n1Sample + (aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND])
                        for nIdx in range(nStart, nEnd) :
                            aSamples[nIdx] = nHandScale * aSamplesRef[nIdx]
                            #print(f'... scale hand {nIdx} : {nHandScale:5.3f} - {aSamplesRef[nIdx]:5.3f} => {aSamples[nIdx]:5.3f}')
                        #print()

                    # Pose scale : +/- 10% -------------------------------------
                    if (len(aVectors) > cPOSE) : # check if pose (idx 2) is taking in count
                        nPoseSizeVar = 0.095
                        nPoseScale = 1.0 + (2 * rd.random() - 1) * nPoseSizeVar

                        for nIdxSample in range(nSample) :
                            nStart = nIdxSample * n1Sample + aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND]
                            nEnd = nIdxSample * n1Sample + aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND] + aShapes[cPOSE]
                            # For all values
                            for nIdx in range(nStart, nEnd) :
                                aSamples[nIdx] = nPoseScale * aSamplesRef[nIdx]
                                #print(f'... scale pose {nIdx} : {nPoseScale:5.3f} - {aSamplesRef[nIdx]:5.3f} => {aSamples[nIdx]:5.3f}')
                            #print()

                    """
                    # Face scale : +/- 6% --------------------------------------
                    if (len(aVectors) > cFACE) : # check (idx cFACE) if face is taking in count
                        nFaceSizeVar = 0.06
                        nFaceScale = 1.0 + (2 * rd.random() - 1) * nFaceSizeVar

                        for nIdxSample in range(nSample) :
                            nStart = nIdxSample * n1Sample + aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND] + aShapes[cPOSE]
                            nEnd = nIdxSample * n1Sample + aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND] + aShapes[cPOSE] + aShapes[cFACE]
                            # For all values
                            for nIdx in range(nStart, nEnd) :
                                aSamples[nIdx] = nFaceScale * aSamplesRef[nIdx]
                                #print(f'... scale face {nIdx} : {nFaceScale:5.3f} - {aSamplesRef[nIdx]:5.3f} => {aSamples[nIdx]:5.3f}')
                            #print()
                    """

                # Hands, pose rotation -----------------------------------------
                if ("rotation" == sDataAugmentation) :
                    # Hands rotation : +/- 15° ---------------------------------
                    nHandRotVar = 15.5 * math.pi / 180
                    nDeltaAngle = (2 * rd.random() - 1) * nHandRotVar

                    for nIdxSample in range(nSample) :
                        nStart = nIdxSample * n1Sample
                        nEnd = nIdxSample * n1Sample + (aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND])
                        # For all point (3 values)
                        for nIdx in range(nStart, nEnd, nDim) :
                            #nLg = math.sqrt(aSamplesRef[nIdx] * aSamplesRef[nIdx] + aSamplesRef[nIdx+1] * aSamplesRef[nIdx+1] + aSamplesRef[nIdx+2] * aSamplesRef[nIdx+2])
                            nLg = math.sqrt(aSamplesRef[nIdx] * aSamplesRef[nIdx] + aSamplesRef[nIdx+1] * aSamplesRef[nIdx+1])
                            nAngle = math.atan2(aSamplesRef[nIdx+1], aSamplesRef[nIdx])
                            
                            # Rotation change sens with left vs right
                            nSign = 1.0 if nIdx < aShapes[cLEFT_HAND] else -1.0
                            #print("... nAngle, nDeltaAngle : ", nAngle*180/math.pi, nDeltaAngle*180/math.pi)
                            
                            aSamples[nIdx] = nLg * math.cos(nAngle + nSign * nDeltaAngle)
                            aSamples[nIdx+1] = nLg * math.sin(nAngle + nSign * nDeltaAngle)
                            # no change for z coordinate
                            
                            #print(f'... rotation hand {nIdx} : {nAngle*180/math.pi:5.1f} + {nSign * nDeltaAngle*180/math.pi:5.1f} - {aSamplesRef[nIdx]:5.3f} => {aSamples[nIdx]:5.3f} / {aSamplesRef[nIdx+1]:5.3f} => {aSamples[nIdx+1]:5.3f}')
                        #print()

                    # Pose rotation : +/- 15° ----------------------------------
                    if (len(aVectors) > cPOSE) :
                        nPoseRotVar = 15.5 * math.pi / 180
                        nDeltaAngle = (2 * rd.random() - 1) * nPoseRotVar

                        for nIdxSample in range(nSample) :
                            nStart = nIdxSample * n1Sample + aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND]
                            nEnd = nIdxSample * n1Sample + aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND] + aShapes[cPOSE]
                            # For all point (3 values)
                            for nIdx in range(nStart, nEnd, nDim) :
                                #nLg = math.sqrt(aSamplesRef[nIdx] * aSamplesRef[nIdx] + aSamplesRef[nIdx+1] * aSamplesRef[nIdx+1] + aSamplesRef[nIdx+2] * aSamplesRef[nIdx+2])
                                nLg = math.sqrt(aSamplesRef[nIdx] * aSamplesRef[nIdx] + aSamplesRef[nIdx+1] * aSamplesRef[nIdx+1])
                                nAngle = math.atan2(aSamplesRef[nIdx+1], aSamplesRef[nIdx])

                                # Rotation change sens with left vs right
                                nSign = 1.0 if ((nIdx % 2) == 1) else -1.0

                                aSamples[nIdx] = nLg * math.cos(nAngle + nSign * nDeltaAngle)
                                aSamples[nIdx+1] = nLg * math.sin(nAngle + nSign * nDeltaAngle)
                            
                                #print(f'... rotation pose {nIdx} : {nAngle*180/math.pi:5.1f} + {nSign * nDeltaAngle*180/math.pi:5.1f} - {aSamplesRef[nIdx]:5.3f} => {aSamples[nIdx]:5.3f} / {aSamplesRef[nIdx+1]:5.3f} => {aSamples[nIdx+1]:5.3f}')
                            #print()
                
                return (aSamples)

            # Data augmentation ------------------------------------------------
            if ("yes" in aDataAugmentation) :

                aSamplesRef = copy.deepcopy(aSamples)

                for nIdxDuplicate in range(nDuplicate) :
                    for sDataAugmentation in aDataAugmentation :

                        if ("yes" == sDataAugmentation) : continue
                        
                        aSamples = mDataAugmented(aSamplesRef, nIdxDuplicate, sDataAugmentation)
                        aLstFeatures.append([sWord] + aSamples)

    sDataFile = sDirModel + sDataFrame

    aColFeat = [str(i).zfill(nDim) for i in range(0, nSample * (aShapes[cLEFT_HAND]+aShapes[cRIGHT_HAND]+aShapes[cPOSE]+aShapes[cFACE]))]
    aCols = ["class"] + aColFeat

    df = pd.DataFrame(aLstFeatures, columns=aCols)
    df.to_csv(sDataFile, index=None)

    nDelay = time.time() - nStartTime
    print("... delay : ", nDelay, nDelay / len(aFileSequences))

# ------------------------------------------------------------------------------
# Load dataset created by features extraction
# ------------------------------------------------------------------------------
def mLsfLoadDataSet(sPathDataFrame, remove_disperse=[]):
    #print(_getframe().f_code.co_name)

    # Get dataframe from file and sort on 'class" column -----------------------
    df = pd.read_csv(sPathDataFrame)
    df = df.sort_values(by = 'class')

    if remove_disperse:
        df = df.drop(remove_disperse, axis=1)

    # Different data group size
    nFeatureHands = 2 * nDim * len(aHandPoints)
    nFeaturePose = nDim * len(aPosePoints)
    nFeatureFace = nDim * len(aFacePoints)
    nFeatureAll = nFeatureHands + nFeaturePose + nFeatureFace
    #print("... ", nFeatureHands, nFeaturePose, nFeatureFace, nFeatureAll, aVectorFeatures)

    # Drop data group not selected ---------------------------------------------
    aColumns = df.columns.tolist()
    for nIdxSample in range(nSample-1, -1, -1) :
        if ('face' not in aVectorFeatures):
            df = df.drop(columns=aColumns[1+nIdxSample*nFeatureAll+nFeatureHands+nFeaturePose:1+nIdxSample*nFeatureAll+nFeatureHands+nFeaturePose+nFeatureFace], axis=1)
        if ('pose' not in aVectorFeatures):
            df = df.drop(columns=aColumns[1+nIdxSample*nFeatureAll+nFeatureHands:1+nIdxSample*nFeatureAll+nFeatureHands+nFeaturePose], axis=1)
        if ('hands' not in aVectorFeatures):
            df = df.drop(columns=aColumns[1+nIdxSample*nFeatureAll:1+nIdxSample*nFeatureAll+nFeatureHands], axis=1)

    print("... dataframe nb lignes / colonnes : ", sPathDataFrame, len(df.axes[0]) , len(df.axes[1]), flush=True)

    aXs = df.drop(["class"], axis=1)
    aYs = df["class"]

    return aXs, aYs

# ------------------------------------------------------------------------------
# Decomposition / reduction dimension with different algorithms
# ------------------------------------------------------------------------------
def mLsfReduction(sDirModel, sDataFrame):
    #print(_getframe().f_code.co_name)

    # Get data normalize
    sPathDataFrame = sDirModel + sDataFrame
    aXInits, aYInits = mLsfLoadDataSet(sPathDataFrame)

    # Normalizer ---------------------------------------------------------------
    for sNormeName, oScaler in aNormalizers.items():
        print("... sNormeName : ", sNormeName) 

        aXs = aXInits.copy(deep=True)
        aYs = aYInits.copy(deep=True)

        # Normalize values
        #aXs = oScaler.fit_transform(aXs.values)

        # Test different decompositions
        for sDecompName, oDecomp in aDecompositions.items():

            print("...... sDecompName : ", sDecompName)

            try :
                aReductFit = oDecomp.fit(aXs, aYs)
                aReduct = aReductFit.transform(aXs)
                aReduct = oScaler.fit_transform(aReduct)

                # Save rule to future decomposition input data
                sFilename = sDirModel + os.path.basename(sDataFrame[:-4]) +'_'+ sDecompName +  '_decom.sav'
                joblib.dump(aReduct, sFilename) 

                #print("... explained_variance_       : ", sDecompName, aReductFit.explained_variance_)
                print(f"... explained_variance_ratio_ : {sDecompName} {sNormeName} {sum(aReductFit.explained_variance_ratio_[:nComponents]):4.3}")
            except :
                print(">>> Decomposition : ", sDecompName)
                continue

            try :
                nScore = round(oDecomp.score(aXs, aYs), 4)
            except :
                nScore = 0.0
            #print("...... nScore : ", nScore)

            if ('yes' in aDisplay) :
                # Find the min and max values to set the scale of the graph
                nXMin, nXMax, nYMin, nYMax, nZMin, nZMax = 1000.0, -1000.0, 1000.0, -1000.0, 1000.0, -1000.0
                for aXY in aReduct :
                    if nXMin > aXY[0] : nXMin = aXY[0]
                    if nXMax < aXY[0] : nXMax = aXY[0]
                    if nYMin > aXY[1] : nYMin = aXY[1]
                    if nYMax < aXY[1] : nYMax = aXY[1]
                    if nZMin > aXY[2] : nZMin = aXY[2]
                    if nZMax < aXY[2] : nZMax = aXY[2]
                #print("... aReduct : ", nXMin, nXMax, nYMin, nYMax, nZMin, nZMax)

                plt.figure(sDecompName + " - " + sNormeName + " - " + str(nScore))

                for nIdx in range(len(aYs)) :

                    # style and color define by the word...
                    sWord = aYs[nIdx].lower()

                    nR = max(0,min(1, (ord(sWord[-1]) - ord('a')) / 26))
                    nG = max(0,min(1, (ord(sWord[0]) - ord('a')) / 26))
                    nB = max(0,min(1, (ord(sWord[len(sWord)//2]) - ord('a')) / 26))

                    text_kwargs = dict(ha='center', va='center', fontsize=7, color=(nR, nG, nB))

                    plt.text(aReduct[nIdx, 0], aReduct[nIdx, 1], aYs[nIdx], **text_kwargs)

                plt.axis([nXMin, nXMax, nYMin, nYMax])
                plt.title(sDecompName + " - " + sNormeName + " - " + str(nScore))

        if ('yes' in aDisplay) : plt.show()

# ------------------------------------------------------------------------------
# Select feature more important with different algorithms
# ------------------------------------------------------------------------------
def mLsfSelection(sDirModel, sDataFrame):
    #print(_getframe().f_code.co_name)

    # Get data normalize
    sPathDataFrame = sDirModel + sDataFrame
    aXInits, aYInits = mLsfLoadDataSet(sPathDataFrame)

    # Normalize between 0..1 (positive value mandatory for certain algorithms)
    oScaler = MinMaxScaler()
    aX = oScaler.fit_transform(aXInits)

    # Select function ----------------------------------------------------------
    for sSelectName, oSelect in aSelectFeatures.items():
        print("... sSelectName : ", sSelectName) 

        # Score function -------------------------------------------------------
        for sScoreName, oScore in aScoreFunctions.items():
            print("...... sScoreName : ", sScoreName) 

            # Instanciate selection feature function ...
            if (sSelectName == "kbest") :
                bestfeatures = oSelect(score_func=oScore, k=nFeature)
            else :
                bestfeatures = oSelect(score_func=oScore)

            # fit to our data
            fit = bestfeatures.fit(aX, aYInits)
            dfscores = pd.DataFrame(fit.scores_)
            dfcolumns = pd.DataFrame(aXInits.columns)

            # concat two dataframes for better visualization 
            featureScores = pd.concat([dfcolumns,dfscores],axis=1)
            featureScores.columns = ['Specs','Score']  #naming the dataframe columns
            aColScore = featureScores.nlargest(nFeature,'Score')

            aCols = list(aColScore['Specs'])
            aScores = list(aColScore['Score'])
            nFeatureHands = len(aHandPoints)
            nFeaturePose = len(aPosePoints)
            nFeatureFace = len(aFacePoints)
            nFeatureVector = sum(aShapes)
            aCoord = ('x', 'y', 'z')

            # Analyse data to display...
            nRang = 0
            print(f"sSelectName;sScoreName;nRang;aCols[nIdx];nFrame;nIdxRel;nIdxFeat;sVector;sCoord;nLandMark;aScores[nIdx]")
            for nIdx in range(nFeature) :

                if nIdx >= nFeaturePrint : break

                nIdxAbs = int(aCols[nIdx])
                # what frame
                nFrame = nIdxAbs // nFeatureVector

                # what part : left / right hand, pose, face
                nIdxRel = nIdxAbs % nFeatureVector
                if nIdxRel >= nDim * (2*nFeatureHands + nFeaturePose) :
                    sVector = "face"
                    nIdxFeat = nIdxRel - nDim * (2*nFeatureHands + nFeaturePose)
                    nLandMark = aFacePoints[nIdxFeat//nDim]
                else :
                    if nIdxRel >= nDim * (2*nFeatureHands) :
                        sVector = "pose"
                        nIdxFeat = nIdxRel - nDim * (2*nFeatureHands)
                        nLandMark = aPosePoints[nIdxFeat//nDim]
                    else :
                        if nIdxRel >= nDim * (nFeatureHands) :
                            sVector = "right"
                            nIdxFeat = nIdxRel - nDim * (nFeatureHands)
                            nLandMark = aHandPoints[nIdxFeat//nDim]
                        else :
                            sVector = "left"
                            nIdxFeat = nIdxRel
                            nLandMark = aHandPoints[nIdxFeat//nDim]
                sCoord = aCoord[nIdxFeat % nDim]          

                nRang += 1
                print(f'{sSelectName};{sScoreName};{nRang};{aCols[nIdx]};{nFrame};{nIdxRel};{nIdxFeat};{sVector};{sCoord};{nLandMark};{aScores[nIdx]:5.1f}')

    return

# ------------------------------------------------------------------------------
# Train and test with different algorithms (classifiers and normalizers)
# ------------------------------------------------------------------------------
def mLsfTrainTest(sDirModel, sDataFrame):
    #print(_getframe().f_code.co_name)

    # Get data normalize
    sPathDataFrame = sDirModel + sDataFrame
    aXInits, aYInits = mLsfLoadDataSet(sPathDataFrame)

    # 20% for test => nValueTest
    nKeyDiff = aYInits.nunique()
    nValueTotal = len(aYInits)
    nValueByKey = int(nValueTotal / nKeyDiff + 0.50)
    #nValueByKey = 30

    nValueTest = 2 * (int(0.20 * nValueByKey + 0.50) // 2)
    nValueTrain = nValueByKey - nValueTest
    nStart = 2 * (rd.randint(0, (nValueByKey - nValueTest)) // 2)   # alea selection test data
    print("... nKeyDiff, nValueTotal, nValueByKey, nValueTrain, nValueTest, nStart : ", nKeyDiff, nValueTotal, nValueByKey, nValueTrain, nValueTest, nStart)

    # Normalizer ---------------------------------------------------------------
    for sNormeName, oScaler in aNormalizers.items():

        aXs = aXInits.copy(deep=True)
        aYs = aYInits.copy(deep=True)

        # Normalize values -----------------------------------------------------
        aXs = oScaler.fit_transform(aXs.values)

        # Save rule to future normalyze input data
        sFilename = sDirModel + os.path.basename(sDataFrame[:-4]) + '_' + sNormeName + '_norm.sav'
        joblib.dump(oScaler, sFilename) 

        # Create train and test set --------------------------------------------
        sYBefore = "---"

        aXTrain = []
        aYTrain = []
        aXTest = []
        aYTest = []

        for aX, sY in zip(aXs, aYs) :

            # Algo 1 : select N first examples / sign --------------------------
            if True :
                if (sYBefore != sY) :
                    nNew = 0
                    nTrain = 0
                    nStart = 2 * (rd.randint(0, (nValueByKey - nValueTest)) // 2)   # alea selection test data

                # bde test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                if (nNew >= nStart) and (nNew < (nStart+nValueTest)) :
                    aXTest.append(aX)
                    aYTest.append(sY)
                else :
                    if nTrain < nValueTrain :
                        aXTrain.append(aX)
                        aYTrain.append(sY)
                        nTrain += 1

                nNew += 1
                sYBefore = sY

            # Algo 2 : select X% examples / sign for test ----------------------
            else :
                #nPourcent = 100 * rd.random()
                nPourcent = rd.randint(0, 100)

                if nPourcent < 20.0 :
                    aXTest.append(aX)
                    aYTest.append(sY)
                else :
                    aXTrain.append(aX)
                    aYTrain.append(sY)

        print("... ----------------------------------------------------------------------------", flush=True)
        print("... sNormeName, len(aYTrain), len(aYTest) : ", sNormeName, len(aYTrain), len(aYTest), flush=True)
        ans = {key: {"score" : []} for key, value in aClassifiers.items()}

        dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(dt, "... Classifying from... " + sPathDataFrame, flush=True)
        nStartTime = time.time()

        # Train ----------------------------------------------------------------
        for sClassifierName, oclassifier in aClassifiers.items():

            # Train our model with train data
            dt = datetime.now().strftime("%Y%m%d-%H%M%S")
            print(dt, "...... Train : ", sClassifierName, flush=True)
            sFilename = sDirModel + os.path.basename(sDataFrame[:-4]) +'_'+ sClassifierName + '_' + sNormeName +  '_clas.sav'
            try :
                oclassifier.fit(aXTrain, aYTrain)

                # Save the model to disk for further use
                joblib.dump(oclassifier, sFilename)

                dt = datetime.now().strftime("%Y%m%d-%H%M%S")
                print(dt, "...... Train done : ", sClassifierName, flush=True)
            except :
                continue

        print("...... Train delay : ", sNormeName, round(time.time() - nStartTime, 1), flush=True)
        nStartTime = time.time()

        # Test -----------------------------------------------------------------
        for sClassifierName, oclassifier in aClassifiers.items():

            # Test our model with train data
            dt = datetime.now().strftime("%Y%m%d-%H%M%S")
            print(dt, "...... Test : ", sClassifierName, flush=True)
            sFilename = sDirModel + os.path.basename(sDataFrame[:-4]) +'_'+ sClassifierName + '_' + sNormeName +  '_clas.sav'
            # Get back the model trained previously
            try :
                oclassifier = joblib.load(sFilename)
            except :
                continue

            # Check confidence with test data
            score = oclassifier.score(aXTest, aYTest)
            ans[sClassifierName]["score"].append(score)

            dt = datetime.now().strftime("%Y%m%d-%H%M%S")
            print(dt, "...... Test done : ", sClassifierName, round(100 * score, 2), flush=True)
        dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(dt, "...... Classification done!", flush=True)
        print("... Test delay : ", sNormeName, round(time.time() - nStartTime, 1), flush=True)
        nStartTime = time.time()

        # Display results ------------------------------------------------------
        print("... ----------------------------------------------------------------------------", flush=True)
        for key, value in aClassifiers.items() :
            sLine = sNormeName +";" + key + ";" + ";".join(str(round(100 * aX, 1)) for aX in ans[key]["score"])
            sLine = sLine.replace('.', ',')
            print(sLine, flush=True)
    return ans

# ------------------------------------------------------------------------------
# 
# ------------------------------------------------------------------------------
def mLsfMostFreq(nMostFreq, aWord, aPredict, bPrint = False) :
    #print(_getframe().f_code.co_name)

    nSize = len(aWord)

    aResult= []
    for nIdx in range(nMostFreq) :
        if (len(aWord)) < 1 : break
        sWordL = max(set(aWord), key = aWord.count)
        nWord = 0
        nProba = 0
        while sWordL in aWord :
            nWord += 1
            nIdxWord = aWord.index(sWordL)
            nProba += aPredict[nIdxWord]
            del aWord[nIdxWord]
            del aPredict[nIdxWord]
    aResult.append((sWordL, nWord))
        #print(f"...... {sWordL:16} {nWord}x {round(100*nProba/nSize,1):4.1f}%")

    ''' aResult = sorted(aResult, key=operator.itemgetter(1, 2), reverse=True)'''

    ''' if bPrint :
        print("... detected :")
        for sWordL in aResult :
            print(f"...... {sWordL}")'''

    return aResult

# ------------------------------------------------------------------------------
# Test the models with real time video capture
# ------------------------------------------------------------------------------
def mLsfInference(sDirModel, sDataFrame, mp_holistic, holistic):
    #print(_getframe().f_code.co_name)

    global sWordDemo, sWordDemoNew, bNextVideo, aDisplResult

    # Initilize video capture
    oCamera, nWidth, nHeight = mLsfInitCamera()

    aSequences = []
    nSequencesInit = 0
    sWord = ""
    nAttemptDemoMax = 2
    nAttemptDemo = 0
    bNextAttempt = True
    while oCamera.isOpened():

        nAttemptDemo += 1

        if sWord == 'au_revoir' : 
            break
        if bNextAttempt :
            print()
            print(f"... sign : {sWordDemo} ... {nAttemptDemo} / {nAttemptDemoMax}")
        bNextAttempt = False

        aSequenceL = []
        aLastFrame = [-1, -1, -1, -1]
        nHandEmpty = 0

        # Get a word...
        nIdxLoop = 0
        while oCamera.isOpened():

            bRetCode, image = oCamera.read()
            if not bRetCode : 
                break
            image = cv2.resize(image, (nWidth, nHeight))

            # Get features from image ------------------------------------------
            bCR, aHandLeft, aHandRight, aPoses, aFaces, bHandLeftAppear, bHandLeftDisAppear, bHandRightAppear, bHandRightDisAppear = mGetFeatures(image, "Test your knowledge (hide your hands between the signs)...")
            if not bCR : 
                break
            #en moyenne 15 trames à ce niveau là
            # Normalize feature : origin middle of shoulders, etc.  ------------
            aHandLeft, aHandRight, aPoses, aFaces = mNormalizeFeatures(aHandLeft, aHandRight, aPoses, aFaces)
            #en moyenne 8 trames à ce niveau là

            #analyser sur 5 et 11 trames en multi-threading 

            # Add default hand when no hand.landmark but pose.wrist ----
            if bExtrapolateHands :
                if (len(aHandLeft) == 0) and (bHandLeftAppear or bHandLeftDisAppear) :
                    aHandLeft = aHandLeftRest
                if (len(aHandRight) == 0) and (bHandRightAppear or bHandRightDisAppear) :
                    aHandRight = aHandRightRest

            # Aggregate pose, and hands features into a unique structure
            # index 0: left hand, index 1: right hand, index 2: pose, index 3: face, 
            aSequenceL.append((aHandLeft, aHandRight, aPoses, aFaces))

            # Interpolate missing features -------------------------------------
            # No interpolation for pose landmarks
            for nIdxVector in (cLEFT_HAND, cRIGHT_HAND) :

                nLength = len(aSequenceL[-1][nIdxVector])
                if nLength > 0 :
                    nLast = aLastFrame[nIdxVector]
                    if nLast < 0 :
                        nLast = nIdxLoop
                        #print(aSequenceL[-1][nIdxVector])

                    elif nLast < (nIdxLoop - 1) :
                        # Interpolation needed
                        # ATTENTION : compteur nHandEmpty compte aussi les interpolations de "face"
                        if bInterpolateHands :
                            nHandEmpty += mInterpolatePart(aLandmarkRef[nIdxVector], nIdxVector, nLast, nLength, nIdxLoop, aSequenceL)

                    aLastFrame[nIdxVector] = nIdxLoop

            nIdxLoop += 1

            # Criteria to transfert frames -------------------------------------
            # No hands on screen
            bNoHands = ((len(aHandLeft) + len(aHandRight)) < 1)
            # Long time beweeen last viewing hands
            nBigHole = 0
            nSeuil = 0
            if aLastFrame[cLEFT_HAND] > -1 :
                nBigHole += nIdxLoop - aLastFrame[cLEFT_HAND]
                nSeuil += 8
            if aLastFrame[cRIGHT_HAND] > -1 :
                nBigHole += nIdxLoop - aLastFrame[cRIGHT_HAND]
                nSeuil += 8
            bBigHole = (nBigHole > nSeuil)
            # At least one hands view
            bViewHand = ((aLastFrame[cLEFT_HAND] + aLastFrame[cRIGHT_HAND]) > -2)
            bBufferFull = (len(aSequenceL) > 18)
            #print(f"... ({bNoHands} and {bBigHole} and {bViewHand}) or ({bBufferFull})")
            
            # Transfert frames to processing
            if (bNoHands and bBigHole and bViewHand) or (bBufferFull) :

                # Suppress sequence features without hands ---------------------
                nSequencesInit = len(aSequenceL)
                for aSequence in aSequenceL :
                    if (len(aSequence[cLEFT_HAND]) != 0) or (len(aSequence[cRIGHT_HAND]) != 0) :
                        aSequences.append(aSequence)

                aSequenceL = []
                nIdxLoop = 0
                aLastFrame = [-1, -1, -1, -1]

                break
                
        # ======================================================================
        
        # Before filtering and calculation, check that we have data to do that...
        #print("... len(aSequences) : ", len(aSequences))
        nSequences = len(aSequences)
        if (nSequences > 1) :
            # Keep landmarks distance of minimal threshold...
            sFile = "camera"
            aSamples, nSequences, nRowFeatures, nFeatures = mLsfFilterFrames(sFile, aSequences, nSequencesInit)
        else :
            # Not enough data...
            #print(".. Not enough data data... : ", nSequences)
            nAttemptDemo -= 1
            continue

        # Check dimension before continuing
        #print("... len(aSamples) ? nSample*sum(aShapes) : ", len(aSamples), nSample * sum(aShapes))
        if (len(aSamples) != (nSample * sum(aShapes))) : continue

        # Keep only data group selected ----------------------------------------
        # Different data group size
        nFeatureHands = 2 * nDim * len(aHandPoints)
        nFeaturePose = nDim * len(aPosePoints)
        nFeatureFace = nDim * len(aFacePoints)
        nFeatureAll = nFeatureHands + nFeaturePose + nFeatureFace

        aSamplesInit = []
        for nIdxSample in range(nSample) :

            if ('hands' in aVectorFeatures):
                nStart = nIdxSample*nFeatureAll
                nEnd = nIdxSample*nFeatureAll+nFeatureHands
                aSamplesInit = aSamplesInit + aSamples[nStart:nEnd]
            if ('pose' in aVectorFeatures):
                nStart = nIdxSample*nFeatureAll+nFeatureHands
                nEnd = nIdxSample*nFeatureAll+nFeatureHands+nFeaturePose
                aSamplesInit = aSamplesInit + aSamples[nStart:nEnd]
            if ('face' in aVectorFeatures):
                nStart = nIdxSample*nFeatureAll+nFeatureHands+nFeaturePose
                nEnd = nIdxSample*nFeatureAll+nFeatureHands+nFeaturePose+nFeatureFace
                aSamplesInit = aSamplesInit + aSamples[nStart:nEnd]

        aWord = []
        aPredict = []
        for sNormeName, oScaler in aNormalizers.items():

            aSamples = copy.deepcopy(aSamplesInit)

            sFilename = sDirModel + os.path.basename(sDataFrame[:-4]) + '_' + sNormeName + '_norm.sav'
            oScaler = joblib.load(sFilename)
            aSamples = oScaler.transform([aSamples])

            # Test -------------------------------------------------------------
            for sClassifierName, oclassifier in aClassifiers.items():

                sFilename = sDirModel + os.path.basename(sDataFrame[:-4]) +'_'+ sClassifierName + '_' + sNormeName +  '_clas.sav'
                # Get back the model trained previously
                try :
                    oclassifier = joblib.load(sFilename)
                except :
                    continue

                # Check confidence with test data
                nPredict = oclassifier.predict_proba(aSamples).max()
                sWord = oclassifier.predict(aSamples)[0]

                aWord.append(sWord)
                aPredict.append(nPredict)

        # Select most frequence proposition ------------------------------------
        aDisplResult = mLsfMostFreq(nMostFreq, aWord, aPredict, bPrint = True)

        # Get N chances before changing sign / word ----------------------------
        if (nAttemptDemo >= nAttemptDemoMax) :
            bNextVideo = True
            nAttemptDemo = 0

        sWordDemo = sWordDemoNew
        bNextAttempt = True
        aSequences = []

    cv2.destroyAllWindows()

# ------------------------------------------------------------------------------
# Continue translation of signs
# input : aSequences
# ------------------------------------------------------------------------------
def mLsfContinue() :
    #print(_getframe().f_code.co_name)
    global aSequences, DirModel, sDataFrame, mp_holistic, holistic
    sWord = ""
    
    
    aWin = (8, 4, 16)
    aWinSkip = (6, 4,12)
    nAttemptThreshold = len(aWin)
    nConfThreshold = 0.18
    
    result = None

    #boucle infinie qui s'arrete en appuyant sur la touche "q"
    while (True) :

        threadmin = Thread(target=mLsfInference(sDirModel, sDataFrame, mp_holistic, holistic)) #premier thread
        threadmax = Thread(target=mLsfInference(sDirModel, sDataFrame, mp_holistic, holistic)) #second thread
        threadprincipal = Thread(target=mLsfInference(sDirModel, sDataFrame, mp_holistic, holistic)) #thread principal qui renvoit le résultat
        #début des thread (lancement)
        threadmin.start()
        threadmax.start() 

        eventmax = Event()
        eventmin = Event()
        threadprincipal.join()


        bCR, aHandLeft, aHandRight, aPoses, aFaces, bHandLeftAppear, bHandLeftDisAppear, bHandRightAppear, bHandRightDisAppear = mGetFeatures(image, "Test your knowledge (hide your hands between the signs)...")
        while (True) :
            time = 0
            #test du temps -> retourne le meilleur résultat
            if (time == 5) :
                eventmin.is_set == True
                resultmin = threadmin
            if (time == 5) :
                eventmax.is_set == True
                resultmax = threadmax
            if eventmin.is_set() :
                resmin = resultmin[1]
            if eventmax.is_set() :
                resmax = resultmax[1]
            if resmax > resmin :
                result = resultmax[0]
            if resmin > resmax :
                result = resultmin[0]


            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #afficher le résultat sur la vidéo au fur et à mesure :
            # langage continu
            time += 0.1
            cv2.namedWindow(sWord)
            (resultheight, resultwidth) = mLsfGetScreenSize()
            cv2.moveWindow(sWord, nWidthScreen//2, 0)    
            nIdxLine = 1
            aDisplResult = mLsfMostFreq(nMostFreq, aWord, aPredict, bPrint = True)
            aDisplay
            for sWordL, number in aDisplResult:
                if (sWordL != phrase[-1]) :
                    if not(bHandRightDisAppear and bHandLeftDisAppear) :
                        cpt += 1
                        phrase.append(sWordL)
                        sSentence = ""
                    if (len(sSentence) > resultwidth ) :
                        phrase.pop(1)
                sSentence = ""
                for e in phrase :
                    sSentence += e + ' '
                cv2.putText(image, f"{(unidecode.unidecode(result))}", (0, 45*nIdxLine), nFont, nFontScale, (0, 0, 255), nThickness, nLineType)
            time += 0.1


# ------------------------------------------------------------------------------
# Play and example of video sign...
# ------------------------------------------------------------------------------
def mLstGetExample(sDirIn, sSrc, sPlayList, lock) :
    #print(_getframe().f_code.co_name)
    
    global sWordDemo, sWordDemoNew, bNextVideo
    
    nWaitTime = 20
    nSlowRate = 2

    """
    nFont = cv2.FONT_HERSHEY_SIMPLEX
    nFontScale = 2
    nThickness = 3
    nLineType = cv2.LINE_AA
    """

    nExampleWidth, nExampleHeight = 0, 0
    
    # Create a named window and put it on location from setting
    sWinName = "Try signing ... as a deaf person"
    cv2.namedWindow(sWinName)
    cv2.moveWindow(sWinName, nExampleWidth, nExampleHeight)    

    for sSrc in aSrc :
        sDirSrc = sDirIn + sSrc + "/"

        # Check each letter list -----------------------------------------------
        aFiles = []
        for sFile in os.listdir(sDirSrc):
            if (not os.path.isfile(sDirSrc + sFile)) : continue
            aFiles.append(sFile)
        nFileMax = len(aFiles)

        nIdxFile = -1
        nIdxFileStep = 20 if nFileMax > 80 else 4
        aPreviousWords = []

        print("... len() {} / {} ".format(nFileLimit, nFileMax))

        while (True) :


            # Rule to select input file ----------------------------------------
            if (sPlayList == "alea") : 
                #nIdxFile = int(nFileMax * rd.random())
                nIdxFile = rd.randint(0, nFileMax)
            else :
                nIdxFile += nIdxFileStep
                #if (nIdxFile >= nFileMax) : break
                if (nIdxFile >= nFileMax) : nIdxFile = 0

            sFile = aFiles[nIdxFile]
            sFileExt = sFile[-4:]
            if (sFileExt != ".mp4") : 
                continue
            

            sBaseFile = os.path.basename(sFile)
            sWordDemoNew = sBaseFile.split("#")[0]
            #print(f"... next word : {nIdxFile} / {nFileMax} => {sWordDemoNew} {sBaseFile}")
            
            # Avoid to propose last N previouw words
            if sWordDemoNew in aPreviousWords : continue
            aPreviousWords.append(sWordDemoNew)
            if len(aPreviousWords) > 6 : del aPreviousWords[0]

            # Play the video several time --------------------------------------
            nIdxVideo = 0
            bNextVideo = False
            while (not bNextVideo) :

                # Open stream on the file --------------------------------------
                camera = cv2.VideoCapture(sDirSrc + sFile)

                while (True) :
                
                    if (sFileExt == ".mp4") :
                        success, image = camera.read()
                        if not success : break
                    image = cv2.resize(image, (nWidth, nHeight))

                    cv2.putText(image, unidecode.unidecode(sWordDemoNew), (0, 45), nFont, nFontScale, (0, 0, 255), nThickness, nLineType)

                    for nIdxSlowRate in range(nSlowRate) :
                        cv2.imshow(sWinName, image)
                        time.sleep(0.020)
                    
                    #cv2.waitKey(nWaitTime)
                    if cv2.waitKey(nWaitTime) == ord('q'): exit()
                    nIdxVideo += 1
                time.sleep(0.4)
                

                #WER and MER method
        
                #thread1 = Threading.Thread(target=mlsgetexample, target=None, name=None, args=, )

# ------------------------------------------------------------------------------
# Which sign maximizes accuracy in inference...
# ------------------------------------------------------------------------------
def mLsfDfAccuracy(sDirModel, sDataFrame) :
    #print(_getframe().f_code.co_name)

    nStartTime = time.time()


    
    # Get data normalize
    sPathDataFrame = sDirModel + sDataFrame
    aXInits, aYInits = mLsfLoadDataSet(sPathDataFrame)
    aXInits = aXInits.to_numpy()
    aYInits = aYInits.to_numpy()

    sPreviousWord = "__start__"
    aWord = []
    aPredict = []

    # Estimate accuracy for all sign in the dataframe --------------------------
    for nIdxSign in range(len(aYInits)) :
    
        sCurrentWord = aYInits[nIdxSign]
        aX = aXInits[nIdxSign]

        # When word change, estimate accuracy
        if (sPreviousWord != sCurrentWord) and (sPreviousWord != "__start__") and (len(aWord) > 0) :

            aScore = []
            aScoreNone = []
            for sWordP, nScore in zip(aWord, aPredict) :
                if sPreviousWord == sWordP :
                    aScore.append(nScore)
                    aScoreNone.append(nScore)
                else :
                    aScore.append(0.0)
            nScoreNoneMean = 100 * statistics.mean(aScoreNone)
            nScoreMean = 100 * statistics.mean(aScore)
            nScoreStd = 100 * statistics.stdev(aScore)
            nScoreMedian = 100 * statistics.median(aScore)
            nFirst = len(aScoreNone)
            nFirstP = 100 * len(aScoreNone) / len(aScore)
            print(f"word;first;meanNone;mean;std;median;nb;{sPreviousWord};{nFirstP:5.2f};{nScoreNoneMean:5.2f};{nScoreMean:5.2f};{nScoreStd:5.2f};{nScoreMedian:5.2f};{nFirst}", flush=True)

            aWord = []
            aPredict = []

        #print("... sPreviousWord, sCurrentWord : ", sPreviousWord, sCurrentWord)
        sPreviousWord = sCurrentWord
        #if sCurrentWord not in ("angleterre","angoisse","anniversaire","an_année_âge","aujourd'hui","aussi","autre","bienvenue","bien_super","commencer","comment","comme_pareil","ils_elles","neiger_neige","ne_pas") : continue

        # Normalizer -----------------------------------------------------------
        for sNormeName, oScaler in aNormalizers.items():

            aSamples = copy.deepcopy(aX)
            aSamples = aSamples.reshape(1, -1)

            sFilename = sDirModel + os.path.basename(sDataFrame[:-4]) + '_' + sNormeName + '_norm.sav'
            oScaler = joblib.load(sFilename)
            aSamples = oScaler.transform(aSamples)

            # Test accuracy ----------------------------------------------------
            for sClassifierName, oclassifier in aClassifiers.items():

                sFilename = sDirModel + os.path.basename(sDataFrame[:-4]) +'_'+ sClassifierName + '_' + sNormeName +  '_clas.sav'
                # Get back the model trained previously
                try :
                    oclassifier = joblib.load(sFilename)
                except :
                    continue

                # Check confidence with test data
                nPredict = oclassifier.predict_proba(aSamples).max()
                sWordPredict = oclassifier.predict(aSamples)[0]

                aWord.append(sWordPredict)
                aPredict.append(nPredict)

        # Select most frequence proposition ------------------------------------
        #aResult = mLsfMostFreq(nMostFreq, aWord, aPredict, bPrint = True)

    nDelay = time.time() - nStartTime
    print(f"... Test delay : {round(nDelay, 1)} s for {len(aYInits)} signs => {round(nDelay / len(aYInits), 1)} s/sign", flush=True)
        
    return

# ------------------------------------------------------------------------------
# Get nature of a word
# ------------------------------------------------------------------------------
aLexique = {}
bLexLoaded = False
def mGetNature(sName) :

    if not bLexLoaded :
        aReader = csv.reader(open('./Lexique383/Lexique.csv', 'r', encoding="utf-8"), delimiter=';')
        for aRow in aReader:
           #print("... aRow : ", aRow)
           sKey, sValue = aRow
           aLexique[sKey] = sValue
    try :
        sNature = aLexique.get(sName.split("_")[0])
    except :
        sNature = 'inc'

    return (sNature)

# ------------------------------------------------------------------------------
# Test the models with real time video capture
# ------------------------------------------------------------------------------
def mLsfAddVideo(mp_holistic, holistic, sDirIn, sSrc) :
    #print(_getframe().f_code.co_name)

    sDirOut = sDirIn + sSrc + '/'
    print("... sDirOut : ", sDirOut)

    # Initialize the camera
    oCamera, nWidth, nHeight = mLsfInitCamera()

    # Use mp4 format with specific fps
    fourcc = cv2.VideoWriter_fourcc(*'mpg4')
    nFPS = 25.0
    sFormat = str(nWidth)

    while (True) :

        # Which sign / word to record
        sRequest = input("... word, #occur, key : ")
        aRequest = sRequest.lower().split(',')
        if len(aRequest) < 2 : break
        time.sleep(3)

        sWord = aRequest[0]
        try :
            nNb = int(aRequest[1])
        except :
            nNb = 1
        if len(aRequest) < 3 :
            sKey = "xxx"
        else :
            sKey = aRequest[2]

        # Get nature of this word
        sNature = mGetNature(sWord)

        for nIdx in range(nNb) :

            sDate = datetime.now().strftime("%Y%m%d-%H%M%S")
            sFilenameVideo = "{}#A#{}#{}#fablab#{}_{}.mp4".format(sWord.replace(' ', '_'), nIdx, sNature, sKey, sDate)

            sTmpFile = './tmp/' + sFilenameVideo

            print("... sTmpFile : ", sTmpFile)
            print("... sign ({}/{}) : ".format(nIdx,nNb))

            aSequences = []
            nHandEmpty = 0
            aLastFrame = [-1, -1, -1, -1]
            bVideo = False 

            oVideoWriter = cv2.VideoWriter(sTmpFile, fourcc, nFPS, (nWidth, nHeight), True)
            if oVideoWriter is None : 
                print(">>> Error oVideoWriter")
                continue

            nFrame = 0
            nIdxLoop = 0
            while True :

                bRetCode, image = oCamera.read()
                if not bRetCode : break
                image = cv2.resize(image, (nWidth, nHeight))

                # Get features from image --------------------------------------
                # Manage display on screen to...
                bCR, aHandLeft, aHandRight, aPoses, aFaces, bHandLeftAppear, bHandLeftDisAppear, bHandRightAppear, bHandRightDisAppear = mGetFeatures(image, "image")
                if not bCR : break

                # Aggregate pose, and hands features into a unique structure
                # index 0: left hand, index 1: right hand
                aSequences.append((aHandLeft, aHandRight))

                # Mark last index frame detected -------------------------------
                for nIdxVector in (cLEFT_HAND, cRIGHT_HAND) :
                    if len(aSequences[-1][nIdxVector]) > 0 :
                        aLastFrame[nIdxVector] = nIdxLoop
                nIdxLoop += 1

                # Criteria to start recording sign -----------------------------
                if aLastFrame[cLEFT_HAND] == -1 and aLastFrame[cRIGHT_HAND] == -1 :
                    continue

                # Criteria to stop recording sign ------------------------------
                # No hands on screen
                bNoHands = ((len(aHandLeft) + len(aHandRight)) < 1)
                # Long time beweeen last viewing hands
                nBigHole = 0
                nSeuil = 0
                if aLastFrame[cLEFT_HAND] > -1 :
                    nBigHole += nIdxLoop - aLastFrame[cLEFT_HAND]
                    nSeuil += 8
                if aLastFrame[cRIGHT_HAND] > -1 :
                    nBigHole += nIdxLoop - aLastFrame[cRIGHT_HAND]
                    nSeuil += 8
                bBigHole = (nBigHole > nSeuil)
                # At least one hands view
                bViewHand = ((aLastFrame[cLEFT_HAND] + aLastFrame[cRIGHT_HAND]) > -2)
                if bNoHands and bBigHole and bViewHand : break

                # Record next frame, and resize it if needed
                oVideoWriter.write(image)

                nFrame += 1
                print("... nFrame : ", nFrame)
                bVideo = True

            if oVideoWriter is not None:
                oVideoWriter.release()
            if (bVideo) : 
                nCR = vc.mFileConvert(sTmpFile, sFormat, sDirOut = sDirOut, bVerbose = False)

                my_file = Path(sTmpFile)
                if my_file.is_file() : os.remove(sTmpFile)

# ------------------------------------------------------------------------------
# main() 
# ------------------------------------------------------------------------------
if __name__ == '__main__' :
    #print("__name__ : ", __name__)

    print("... start date : ", datetime.now().strftime("%Y%m%d-%H%M%S"))
    nArg = len(sys.argv)

    # Get screen size
    #nWidthScreen, nHeightScreen = mLsfGetScreenSize()

    # Get user's parameters or default values
    #---------------------------------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--in",     default="./select/", help="input files directory")
    ap.add_argument("-m", "--mod",    default="./model/", help="output / input dataframe / model directory")
    ap.add_argument("-s", "--src",    default="lsf-xxxx", help="video source (dicoelix, pisourd, sematos, other, lsf-xxx, test)")
    ap.add_argument("-e", "--set",    default="directory", help="video set : directory, array, test")
    ap.add_argument("-n", "--nb",     default="0", help="how many video, 0 = all")
    ap.add_argument("-p", "--play",   default="sequence", help="video playlist : alea, sequence")
    ap.add_argument("-t", "--dis",    default="hands", help="display video on screen : hands, pose, face, yes, no")
    ap.add_argument("-d", "--df",     default="lsf-xxxx_3_hp_8.npy", help="dataframe name")
    ap.add_argument("-c", "--cfg",    default="medium", help="configuration classifier x normalizer: small, medium, large, all")
    ap.add_argument("-a", "--action", default="infe", help="vide[o], extr[action], spli[t], feat[ure], redu[ction], sele[ction], trai[n&test], accu[racy], infe[rence], demo[nstration]")
    ap.add_argument("-g", "--aug",    default="no", help="data augmentation : yes, no, scale, rotation, shift, noise")
    ap.add_argument("-y", "--naug",   default="1", help="data augmentation : how many duplicate values")
    ap.add_argument("-v", "--vec",    default="hands,pose", help="vector features composed of: hands[,pose[,face]]")
    ap.add_argument("-x", "--xpol",   default="x", help="[i]nterpolate and/or [e]xtrapolate missing trames")
    ap.add_argument("-z", "--dim",    default="3", help="vector features dimension 2 (x,y) or 3 (x,y,z)")
    ap.add_argument("-f", "--sample", default="8", help="number of frames by feature vectors")
    ap.add_argument("-r", "--attempt",default="1", help="number of attempt when no landmark")
    args = vars(ap.parse_args())

    bParamError = False
    # Initialize parameters ----------------------------------------------------
    sDirIn = args['in']
    sDirModel = args['mod']
    sSrc = args['src'].lower()
    aSrc = sSrc.split(",")
    for s1Src in aSrc :
        sPath = sDirIn + s1Src
        if not os.path.isdir(sPath) :
            bParamError = True
            print(">>> src    : error value")
    sSet = args['set'].lower()
    if sSet not in ('directory', 'array', 'test') : 
        bParamError = True
        print(">>> set    : error value")
    nFileLimit = int(args['nb'])
    if (nFileLimit < 1) : nFileLimit = 9999999999
    sPlayList = args['play'].lower()
    if sPlayList not in ('alea', 'sequence') : 
        bParamError = True
        print(">>> play    : error value")

    sDisplay = args['dis'].lower()
    aDisplay = sDisplay.split(",")
    for s1Dis in aDisplay :
        if s1Dis not in ('hands', 'pose', 'face', 'yes', 'no') : 
            bParamError = True
            print(">>> dis    : error value")

    if ("hands" in aDisplay or "pose" in aDisplay or "face" in aDisplay) :
        aDisplay.append('yes')
        nFPS = 25
    else :
        nFPS = 200
    nWaitTime = int(1000 / 2 / nFPS)
    nWaitTime = 1

    sCfg = args['cfg'].lower()
    if sCfg not in ('small', 'medium', 'large', 'all') : 
        bParamError = True
        print(">>> cfg : error value")
    if (sCfg == 'small') :
        aNormalizers = aNormalizers_small
        aClassifiers = aClassifiers_small
        aDecompositions = aDecompositions_small
    elif (sCfg == 'medium') :
        aNormalizers = aNormalizers_medium
        aClassifiers = aClassifiers_medium
        aDecompositions = aDecompositions_medium
    elif (sCfg == 'large') :
        aNormalizers = aNormalizers_large
        aClassifiers = aClassifiers_large
        aDecompositions = aDecompositions_large
    elif (sCfg == 'all') :
        aNormalizers = aNormalizers_all
        aClassifiers = aClassifiers_all
        aDecompositions = aDecompositions_all

    sDataFrame = args['df']
    sAction = args['action'].lower()[0:4]
    if sAction not in ('vide', 'extr', 'spli', 'feat', 'redu', 'sele', 'trai', 'accu', 'infe', 'demo', 'all*', 'test') : 
        bParamError = True
        print(">>> action : error value")

    sDataAugmentation = args['aug'].lower()
    aDataAugmentation = sDataAugmentation.split(',')
    if ("yes" in aDataAugmentation) :
        aDataAugmentation.append("rotation")
        aDataAugmentation.append("scale")
        aDataAugmentation.append("shift")
        aDataAugmentation.append("noise")
    if ("scale" in aDataAugmentation) or ("rotation" in aDataAugmentation) or ("shift" in aDataAugmentation) or ("noise" in aDataAugmentation) :
        aDataAugmentation.append("yes")
    nDuplicate = int(args['naug'])

    # Which features : hands[,pose[,face]]
    # Threshold between 2 frames, which we estimate as significant for detecting motion (Cf. xlsx / dist feat.)
    nDim = int(args['dim'])
    if nDim == 2 :
        aDistThreshold = [0.0850, 0.0850, 0.110, 0.0850]
        aDistThreshold = [0.0630, 0.0630, 0.0768, 0.0598]
    else :
        aDistThreshold = [0.341, 0.341, 0.752, 0.288] # 3 frames/sign = => 10,5  frames
        aDistThreshold = [0.371, 0.371, 0.852, 0.323] # 4 frames/sign = =>  9,97 frames
        aDistThreshold = [0.399, 0.399, 0.952, 0.359] # 5 frames/sign = =>  9,46 frames
        aDistThreshold = [0.431, 0.431, 1.062, 0.394] # 6 frames/sign = =>  8,97 frames
        aDistThreshold = [0.461, 0.461, 1.179, 0.434] # 7 frames/sign = =>  8,55 frames
        aDistThreshold = [0.489, 0.489, 1.298, 0.478] # 8 frames/sign = =>  8,17 frames

        aDistThreshold = [0.371, 0.371, 0.852, 0.323] # 4 frames/sign = =>  9,97 frames

    nSample = 8
    nSample = int(args['sample']) # number frame by word
    if (25 < nSample) or (nSample < 4) : nSample = 8
    print("... nSample, aDistThreshold : ", nSample, aDistThreshold)
        
    sVectorFeatures = args['vec']
    aVectorFeatures = sVectorFeatures.split(',')
    aVectors = [cLEFT_HAND, cRIGHT_HAND, cPOSE, cFACE]
    aShapes = [nDim*len(aHandPoints), nDim*len(aHandPoints), nDim*len(aPosePoints), nDim*len(aFacePoints)]

    sRefXpolate = "wrist"     # wrist, linear
    print("... sRefXpolate, aVectors, aShapes       : ", sRefXpolate, aVectors, aShapes)

    # bde test - Inter and extra-polate missing hand features ------------------
    # used in : mLsfFeature, mLsfInference
    sXpolate = args['xpol'].lower()
    bInterpolateHands = ('i' in sXpolate)
    bExtrapolateHands = ('e' in sXpolate)
    print("... bInterpolateHands, bExtrapolateHands : ", bInterpolateHands, bExtrapolateHands)

    # Attempt X time, when no hands detected -----------------------------------
    nAttemptFeat = int(args['attempt'])        # 2: 90, 3:40, 4:30, 5:25  <<< bde test <<<<<<<<<<<<<<<<
    if (4 < nAttemptFeat) or (nAttemptFeat < 1) : nAttemptFeat = 1
    nAttemptThreshold = (1 - math.exp(math.log(0.5) / nAttemptFeat)) # After X attempts, we want 0.5 probability
    print("... nAttemptFeat, nAttemptThreshold      : ", nAttemptFeat, nAttemptThreshold)
    
    if (nArg == 1) or bParamError :
        print("usage : ")
        print("      --in     default='./select/', input files directory")
        print("      --mod    default='./model/', output / input dataframe / model directory")
        print("      --src    default='lsf-xxxx', video source (dicoelix, pisourd, sematos, other, lsf-xxx, test)")
        print("      --set    default='directory', video set : directory, array, test")
        print("      --nb     default='0', how many video, 0 = all")
        print("      --play   default='sequence', video playlist : alea, sequence")
        print("      --dis    default='hands', display video on screen : hands, pose, face, yes, no")
        print("      --df     default='lsf-xxxx_3_hp_8.npy', dataframe name")
        print("      --cfg    default='medium', configuration classifier x normalizer: small, medium, large, all")
        print("      --action default='infe', what action to do : vide[o], extr[action], spli[t], feat[ure], redu[ction], sele[ction], trai[n&test], accu[racy], infe[rence], demo[nstration], test[continue]")
        print("      --aug    default='no', data augmentation : scale, rotation, shift, noise, yes, no")
        print("      --naug   default='1', data augmentation : how many duplicate values")
        print("      --vec    default='hands,pose', vector features composed of: hands[,pose[,face]]")
        print("      --xpol   default='x', [i]nterpolate and/or [e]xtrapolate missing trames, other letter no xpolation")
        print("      --dim    default='3', vector features dimension 2 (x,y) or 3 (x,y,z)") 
        print("      --sample default='8', number of frames by feature vectors")
        print("      --attempt,default='1', number of attempt when no landmark")
        print("... at least one argument needed")
        print()
        print(f"... python {sys.argv[0]} --in {sDirIn} --mod {sDirModel} --set {sSet} --nb {nFileLimit} --play {sPlayList} --aug {sDataAugmentation} --naug {nDuplicate} --dis {sDisplay} --vec {sVectorFeatures} --dim {nDim} --attempt {nAttemptFeat} --sample {nSample} --xpol {sXpolate} --cfg {sCfg} --src {sSrc} --df {sDataFrame} --action {sAction} > lsf-xxxx_3_hp_x_Flip_8.txt")
        print()

        exit(1)

    print(f"... python {sys.argv[0]} --in {sDirIn} --mod {sDirModel} --set {sSet} --nb {nFileLimit} --play {sPlayList} --aug {sDataAugmentation} --naug {nDuplicate} --dis {sDisplay} --vec {sVectorFeatures} --dim {nDim} --attempt {nAttemptFeat} --sample {nSample} --xpol {sXpolate} --cfg {sCfg} --src {sSrc} --df {sDataFrame} --action {sAction} > lsf-xxxx_3_hp_x_Flip_8.txt")

    # package versions ---------------------------------------------------------
    print("... Package versions")
    print("...... csv        : ", csv.__version__)
    print("...... openCV     : ", cv2.__version__)
    print("...... joblib     : ", joblib.__version__)
    print("...... matplotlib : ", matplotlib.__version__)
    print("...... numpy      : ", np.__version__)
    print("...... pandas     : ", pd.__version__)
    print("...... sklearn    : ", sklearn.__version__)

    # Initialize MediaPipe package ---------------------------------------------
    mp_drawing, mp_drawing_styles, mp_holistic, holistic = mLsfInitMediaPipe(sAction)

    if (sAction == "vide") :
        # Get video sequences of signs
        ans = mLsfAddVideo(mp_holistic, holistic, sDirIn, sSrc)

    if (sAction == "extr") :
        # Extract features from sign videos
        ans = mLsfRawFeature(sDirIn, sSrc, sDirModel, sDataFrame, mp_holistic, holistic)

    if (sAction == "spli") :
        # Split extracted features into differents dataset
        ans = mLsfSplit(sDirIn, sSrc, sDirModel)

    if (sAction == "feat") :
        # Extract features from sign videos
        ans = mLsfFeature(sDirIn, sSrc, sDirModel, sDataFrame, mp_holistic, holistic)

    if (sAction == "redu") :
        # Analyse features : looking for Principal Component Analysis ...
        ans = mLsfReduction(sDirModel, sDataFrame)

    if (sAction == "sele") :
        # Analyse features : select most relevant features
        ans = mLsfSelection(sDirModel, sDataFrame)

    if (sAction == "trai") :
        # Use videos dataset to train some models
        ans = mLsfTrainTest(sDirModel, sDataFrame)

    if (sAction == "accu") :
        # Test all video signs to identify their inference accuracy...
        mLsfDfAccuracy(sDirModel, sDataFrame)

    if (sAction == "all*") :
        # Extract features from sign videos
        ans = mLsfFeature(sDirIn, sSrc, sDirModel, sDataFrame, mp_holistic, holistic)

        # Use videos dataset to train some models
        ans = mLsfTrainTest(sDirModel, sDataFrame)

        # Test all video signs to identify their inference accuracy...
        mLsfDfAccuracy(sDirModel, sDataFrame)

        # Analyse features : select most relevant features
        ans = mLsfSelection(sDirModel, sDataFrame)

    if (sAction == "infe") :
        # Do prediction from real time video
        ans = mLsfInference(sDirModel, sDataFrame, mp_holistic, holistic)

    if (sAction == "test") :
        #test : do prediction from real time and make a sentence at the end
        lock = threading.Lock() 
        oThreadVideo = threading.Thread(target=mLstGetExample, args=(sDirIn, sSrc, sPlayList, lock))
        oThreadVideo.daemon = True
        oThreadVideo.start()
        ans = mLsfContinue()

    if (sAction == "demo") :
        # Do prediction from real time video with example given to user

        # Start the thread showing video examples
        lock = threading.Lock() 
        oThreadVideo = threading.Thread(target=mLstGetExample, args=(sDirIn, sSrc, sPlayList, lock))
        oThreadVideo.daemon = True
        oThreadVideo.start()

        # Main code used to verify your attempts
        ans = mLsfInference(sDirModel, sDataFrame, mp_holistic, holistic)