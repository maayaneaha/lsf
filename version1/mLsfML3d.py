# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

# import the necessary packages ------------------------------------------------

import argparse
import copy
import csv
import cv2
import operator
import joblib
import math
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import random as rd
import screeninfo as sc
import sklearn
import sys
import threading
import time

import mVideoConvert as vc

from text_to_speech import speak
from pathlib import Path
from sys import _getframe
from datetime import datetime

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler, QuantileTransformer, PowerTransformer, MaxAbsScaler
from sklearn.model_selection import train_test_split as data_split

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

# List of scalers to normalize data --------------------------------------------
aNormalizers = {
    "Identity"  : cLsfNormalizerIdentity(),
    "MaxAbs"    : MaxAbsScaler(), 
    "MinMax"    : MinMaxScaler(), 
    "Normalize" : Normalizer(), 
    "Power"     : PowerTransformer(), 
    "Quantile"  : QuantileTransformer(n_quantiles=1000), 
    "Robust"    : RobustScaler(quantile_range=(17.0, 83.0)), 
    "Standard"  : StandardScaler()
}

aNormalizers = {
    "MaxAbs"    : MaxAbsScaler(), 
    "Power"     : PowerTransformer(), 
    "Quantile"  : QuantileTransformer(n_quantiles=100), 
    "Standard"  : StandardScaler()
}

aNormalizers = {
    "Power"     : PowerTransformer(), 
    "Standard"  : StandardScaler()
}

aNormalizers = {
    "MaxAbs"    : MaxAbsScaler(), 
    "Power"     : PowerTransformer(), 
    "Quantile"  : QuantileTransformer(n_quantiles=100), 
    "Standard"  : StandardScaler()
}

aNormalizers = {
    "MaxAbs"    : MaxAbsScaler(), 
    "Power"     : PowerTransformer(), 
    "Quantile"  : QuantileTransformer(n_quantiles=100), 
    "Robust"    : RobustScaler(quantile_range=(17.0, 83.0)), 
    "Standard"  : StandardScaler()
}

# List of classifiers ----------------------------------------------------------
aClassifiers = {
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
    "SVM_linear" : SVM(kernel='linear', gamma='scale', C=1, cache_size=4000),
    "SVM_poly"   : SVM(kernel='poly', degree=3, gamma='scale', C=1, cache_size=4000),
    "SVM-RBF"    : SVM(kernel='rbf', gamma='scale', C=1, cache_size=4000)
}

aClassifiers = {
    "LDA"        : LDA(),
    "MLP"        : MLP(alpha=1, max_iter=600, batch_size='auto'),
    "QDA"        : QDA(),
}

aClassifiers = {
    "LDA"        : LDA(),
    "MLP"        : MLP(alpha=1, max_iter=600, batch_size='auto'),
}
aClassifiers = {
    "KN_ball"    : KN(n_neighbors=5, weights='uniform', algorithm='ball_tree'),
    "LDA"        : LDA(),
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
aDecompositions = {
    "PCA"               : PCA(n_components=nComponents), 
    "FastICA"           : FastICA(n_components=nComponents), 
    "FactorAnalysis"    : FactorAnalysis(n_components=nComponents), 
    "IPCA"              : IncrementalPCA(n_components=nComponents),
    "KernelPCA"         : KernelPCA(n_components=nComponents),
    "LatentDirichlet"   : LatentDirichletAllocation(n_components=nComponents),
    "MiniBatchDict"     : MiniBatchDictionaryLearning(n_components=nComponents),
    "MiniBatchSpar"     : MiniBatchSparsePCA(n_components=nComponents),
    "NMF"               : NMF(n_components=nComponents),
    "SparsePCA"         : SparsePCA(n_components=nComponents),
    "TruncatedSVD"      : TruncatedSVD(n_components=nComponents),
    "LDA"               : LDA(n_components=nComponents),
}

aDecompositions = {
    "LDA"               : LDA(n_components=nComponents),
    "PCA"               : PCA(n_components=nComponents), 
}

# Global variables -------------------------------------------------------------

nWidth, nHeight = 800, 600

nSkipPourcentage = 0.05     # 15% or 5% of frame ignore at begining and end of record
nMostFreq = 5               # number of prédiction word to display

nConfidence = 0.50

# Landmark used in hands, pose, face mdels
aHandPoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#aPosePoints = [0, 11, 12, 13, 14, 15, 16, 19, 20]
#cP_NOSE, cP_LEFT_SHOULDER, cP_RIGHT_SHOULDER, cP_LEFT_ELBOW, cP_RIGHT_ELBOW, cP_LEFT_WRIST, cP_RIGHT_WRIST, cP_LEFT_INDEX, cP_RIFHT_INDEX = 0, 1, 2, 3, 4, 5, 6, 7, 8
aPosePoints = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
cP_NOSE, cP_LEFT_SHOULDER, cP_RIGHT_SHOULDER, cP_LEFT_ELBOW, cP_RIGHT_ELBOW, cP_LEFT_WRIST, cP_RIGHT_WRIST, cP_LEFT_PINKY, cP_RIFHT_PINKY, cP_LEFT_INDEX, cP_RIFHT_INDEX, cP_LEFT_THUMB, cP_RIFHT_THUMB = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
aFacePoints =  [19, 13, 14, 33, 263, 53, 283, 55, 285, 76, 306, 124, 353, 133, 362, 145, 374, 159, 386, 213, 433]
#cF_NOSE = 19
cF_NOSE = 0

# What group of landmarks we use : (0,1) = hands, (0, 1, 2) = hands + pose, (0, 1, 2, 3) = hands + pose + face
cLEFT_HAND = 0
cRIGHT_HAND = 1
cPOSE = 2
cFACE = 3

aLandmarkRef = [cP_LEFT_WRIST, cP_RIGHT_WRIST, 0, cP_NOSE]

sWordDemo = ""
sWordDemoNew = ""
bNextVideo = True

# ------------------------------------------------------------------------------
# Get screen size
# ------------------------------------------------------------------------------
def mLsfGetScreenSize() :
    #print(_getframe().f_code.co_name)

    # get the size of the screen
    for nIdxScreen in range(8) :
        try :
            oScreen = sc.get_monitors()[nIdxScreen]
        except :
            continue
        nWidth, nHeight = oScreen.width, oScreen.height    
        print("... nIdxScreen, nWidth, nHeight : ", nIdxScreen, nWidth, nHeight)
    nWidth = nWidth // 2
    nHeight = nHeight // 2
    
    return nWidth, nHeight

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

def mPointDistance(aA, aB) :
    #print(_getframe().f_code.co_name)

    if (nDim == 2) : nDist = np.sqrt((aA[0] - aB[0])**2 + (aA[1] - aB[1])**2)
    if (nDim == 3) : nDist = np.sqrt((aA[0] - aB[0])**2 + (aA[1] - aB[1])**2 + (aA[2] - aB[2])**2)
    return nDist

def mPointNormalize(aPoints, aOrigin, nRefW, nRefH, nRefD, nDilate = 1.0) :
    #print(_getframe().f_code.co_name)

    for nIdxPoint in range(len(aPoints)) :
        if (nDim == 2) : 
            aPoints[nIdxPoint] = (nDilate * (aPoints[nIdxPoint][0] - aOrigin[0]) / nRefW, 
                                  nDilate * (aPoints[nIdxPoint][1] - aOrigin[1]) / nRefH)
        if (nDim == 3) : 
            aPoints[nIdxPoint] = (nDilate * (aPoints[nIdxPoint][0] - aOrigin[0]) / nRefW, 
                                  nDilate * (aPoints[nIdxPoint][1] - aOrigin[1]) / nRefH,
                                  nDilate * (aPoints[nIdxPoint][2] - aOrigin[2]) / nRefD)
    return aPoints

# ------------------------------------------------------------------------------
# Initilize mediapipe models : hands, pose, face
# ------------------------------------------------------------------------------
def mLsfInitMediaPipe() :
    #print(_getframe().f_code.co_name)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_faceMesh = mp.solutions.face_mesh

    #hands = mp_hands.Hands(static_image_mode=False, model_complexity=0, min_detection_confidence=0.50, min_tracking_confidence=0.50)
    #pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=nConfidence, min_tracking_confidence=nConfidence)
    pose = mp_pose.Pose(min_detection_confidence=nConfidence, min_tracking_confidence=nConfidence)
    faceMesh = mp_faceMesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=nConfidence, min_tracking_confidence=nConfidence)

    return mp_drawing, mp_drawing_styles, mp_hands, mp_pose, mp_faceMesh, hands, pose, faceMesh 

# ------------------------------------------------------------------------------
# Interpolate coordinates between 2 known positions of hands or face
# ------------------------------------------------------------------------------
def mInterpolatePart(cLM, nIdxVector, nLast, nLong, nIdxLoop, aSequences) :       
    #print(_getframe().f_code.co_name)

    nEmpty = 0
    for nIdxMissing in range(nLast + 1, nIdxLoop) :

        # Proportionality coefficients
        if (sInterPolation == "linear") or (nIdxVector == cFACE):
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

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try :
        handsResults = hands.process(image)
        poseResults = pose.process(image)
        faceResults = faceMesh.process(image)
    except :
        return False, None, None, None, None
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    aHandLeft = []
    aHandRight = []
    aPoses = []
    aFaces = []

    # Get and draw the hands annotations on the image ---------------------------
    if handsResults.multi_hand_landmarks:

        aHandsType = []
        aHands = []
        #print("... multi_hand_landmarks : ", handsResults.multi_hand_landmarks)
        for handLandMarks in handsResults.multi_hand_landmarks :

            if ("hands" in aDisplay) :
                mp_drawing.draw_landmarks(
                    image,
                    handLandMarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            myHand = []
            for lm in handLandMarks.landmark:
                #myHand.append((int(lm.x*width),int(lm.y*height)))
                if (nDim == 2) : myHand.append((lm.x, lm.y))
                if (nDim == 3) : myHand.append((lm.x, lm.y, lm.z))
            aHands.append(myHand)
            #aHands = np.append(aHands, myHand)

        # 21 hands landmarks ----------------------------------------------------
        for hand in handsResults.multi_handedness:

            handType=hand.classification[0].label
            aHandsType.append(handType.lower())

        # Split landmark into left and right hands
        for sHandType, aHandLandMark in zip(aHandsType, aHands) :
            # Seem left and right are inversed
            if sHandType == "left" :
                aHandRight = aHandLandMark
            else :
                aHandLeft = aHandLandMark

    # Get and draw the face annotations on the image ---------------------------
    if faceResults.multi_face_landmarks:

        for faceLandMarks in faceResults.multi_face_landmarks:

            if ("face" in aDisplay) :
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=faceLandMarks,
                    connections=mp_faceMesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                """
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=faceLandMarks,
                    connections=mp_faceMesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                """
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=faceLandMarks,
                    connections=mp_faceMesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

        for faceLandMarks in faceResults.multi_face_landmarks :
            for nIdxFace, lm in enumerate(faceLandMarks.landmark):
                # Peut-être pas les bons points... le total est différent...
                if (nIdxFace in aFacePoints) :
                    #aFaces.append((int(lm.x*width),int(lm.y*height)))
                    if (nDim == 2) : aFaces.append((lm.x, lm.y))
                    if (nDim == 3) : aFaces.append((lm.x, lm.y, lm.z))

    # Get and draw the pose annotation on the image ----------------------------
    if ("pose" in aDisplay) :
        mp_drawing.draw_landmarks(
            image,
            poseResults.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # Keep 9/33 upper body pose landmarks --------------------------------------
    try :
        for nIdxPose, lm in enumerate(poseResults.pose_landmarks.landmark) :

            if (nIdxPose in aPosePoints) :
                #aPoses.append((int(lm.x*width),int(lm.y*height)))
                if (nDim == 2) : aPoses.append((lm.x, lm.y))
                if (nDim == 3) : aPoses.append((lm.x, lm.y, lm.z))

    except :
        return False, None, None, None, None

    if ("hands" in aDisplay) or ("pose" in aDisplay) or ("face" in aDisplay) :
    
        cv2.namedWindow(sWord)
        cv2.moveWindow(sWord, nWidthScreen//2, 0)    

        cv2.imshow(sWord, image)
        if cv2.waitKey(nWaitTime) & 0xFF == 27: pass

    return True, aHandLeft, aHandRight, aPoses, aFaces

# ------------------------------------------------------------------------------
# Normalize features try to get features independent from distance and people
# ------------------------------------------------------------------------------
def mNormalizeFeatures(aHandLeft, aHandRight, aPoses, aFaces) :
    #print(_getframe().f_code.co_name)

    # Normalisation : origin middle of shoulders -------------------------------
    # width  scale: nRefW = 1/3 = shoulder to shoulder
    # height scale: nRefH = 1/4 = nose to middle of shoulders
    aOrigin = mPointCenter(aPoses[cP_LEFT_SHOULDER], aPoses[cP_RIGHT_SHOULDER])
    nRefW = mPointDistance(aPoses[cP_LEFT_SHOULDER], aPoses[cP_RIGHT_SHOULDER])
    nRefH = mPointDistance(aPoses[cP_NOSE], aOrigin)

    # bde test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    if False :
        nRefL = mPointDistance(aHandLeft[0], aHandLeft[10])
        nRefR = mPointDistance(aHandRight[0], aHandRight[10])
        print (f"nRefW;nRefH;nRefL;nRefR;{round(nRefW, 5):6.5f};{round(nRefH, 5):6.5f};{round(nRefL, 5):6.5f};{round(nRefR, 5):6.5f}")

    # Use relative sizes -------------------------------------------------------
    aPoses = mPointNormalize(aPoses, aOrigin, nRefW, nRefH, nRefW)
    #print("... nose, aOrigin, nRefW, nRefH : ", aPoses[0], aOrigin, nRefW, nRefH)

    aFaces = mPointNormalize(aFaces, aOrigin, nRefW, nRefH, nRefW)

    # For hands, use wrist as origin after normalisation vs pose
    if (len(aHandLeft) > 0) :
        aHandLeft = mPointNormalize(aHandLeft, aHandLeft[0], nRefW, nRefH, nRefW)
        #print("... aHandLeft : ", aHandLeft[1])

    if (len(aHandRight) > 0) :
        aHandRight = mPointNormalize(aHandRight, aHandRight[0], nRefW, nRefH, nRefW)
        #print("... aHandRight : ", aHandRight[1])

    return aHandLeft, aHandRight, aPoses, aFaces

# ------------------------------------------------------------------------------
# Keep frames when its distance from previous one is > threshold
# Select or expand to keep just nSample samples
# ------------------------------------------------------------------------------
def mLsfFilterFrames(sFile, aSequences) :
    #print(_getframe().f_code.co_name)

    aFeatures = []
    nSequences = len(aSequences)
    bThreshold = False

    # Keep first and last landmark, and when hands or pose or face move > threshold
    aFeatures.append(aSequences[0])
    for nIdxFrame in range(1, nSequences) :
        aDistance = mGetDistance(aFeatures[-1], aSequences[nIdxFrame])

        # Keep this frame if at least one distance is greater than threshold
        # Only the distance of the hands is important
        bThreshold = False
        # bde test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #for nIdxVector in aVectors[0:2] :  # hands
        #for nIdxVector in aVectors[2:3] :  # pose
        for nIdxVector in aVectors[0:3] :  # hands + pose
            if aDistance[nIdxVector] > aDistThreshold[nIdxVector] :
                bThreshold = True
                break
        if bThreshold :
            aFeatures.append(aSequences[nIdxFrame])
    if not bThreshold : aFeatures.append(aSequences[-1])

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
    if False :
        print("... sFile, nSequences, nRowFeatures, nFeatures, nSample, nOffSet, nStep, len() : ", sFile, nSequences, nRowFeatures, nFeatures, nSample, nOffSet, nStep, len(aSamples))

    return aSamples, nSequences, nRowFeatures, nFeatures

# ------------------------------------------------------------------------------
# Manage feature extraction
# ------------------------------------------------------------------------------
def mLsfFeature(sDirIn, sDirModel, sDataFrame, mp_hands, mp_pose, mp_faceMesh, hands, pose, faceMesh) :
    #print(_getframe().f_code.co_name)

    nStartTime = time.time()
    aLstFeatures = []

    for sSrc in aSrc :
        sDirSrc = sDirIn + sSrc + "/"

        # Check each letter list -----------------------------------------------
        aFiles = []
        if (sSet == "directory") :

            for sFile in os.listdir(sDirSrc):
                if (not os.path.isfile(sDirSrc + sFile)) : continue
                aFiles.append(sFile)

        elif (sSet == "test") :

            # Liste files signs don't work with crop process
            aFiles = [
                "vent#A#0#nmm#dicoelix#vent-2.mp4",
                "manger#A#0#ver#dicoelix#manger_v_2_6.mp4",
                "cache#A#0#nmm#dicoelix#cache_nm_10_6.mp4", 
            ]

        else :

            # Liste files signs with major difference...
            if sSrc == "pisourd" :
                aFiles = [
                    "amoureuse#A#0#nmf#pisourd#VideoStream.php%3Fid=167.mp4",
                    "amoureux#A#0#nmm#pisourd#VideoStream.php%3Fid=168.mp4",
                    "attirant#A#0#adj#pisourd#VideoStream.php%3Fid=282.mp4",
                ]

            if sSrc == "sematos" :
                aFiles = [
                    "arriver#A#0#ver#sematos#aZlmZlslashIods%3D.mp4",
                    "bas_[collant]#A#0#nom#sematos#Z5psZVslashIods%3D.mp4",
                    "calculer#A#0#ver#sematos#Z5NsY1slashIods%3D.mp4",
                ]

            if sSrc == "dicoelix" :
                aFiles = [
                    "atypique#A#0#adj#dicoelix#atypique_adj_2_6-2.mp4",
                    "aimer#A#0#ver#dicoelix#aimer-2.mp4",
                    "argent#A#0#nmm#dicoelix#argent-2.mp4",
                ]

        nStartTime = time.time()

        nIdxFile = -1
        nFileMax = len(aFiles)
        nFile = 0

        print("... len() {} / {} ".format(nFileLimit, nFileMax))

        while (True) :

            # Rule to select input file ----------------------------------------
            if (sChoice == "alea") : 
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
            if False : print("... sBaseFile : ", sBaseFile, nIdxFile)

            for sMode in ('original', 'flip') :

                # Open stream on the file ------------------------------------------
                if (sFileExt == ".mp4") :
                    camera = cv2.VideoCapture(sDirSrc + sFile)
                    sCR = camera.set(cv2.CAP_PROP_FPS, nFPS)
                    #print("... FPS : ", camera.get(cv2.CAP_PROP_FPS))

                aSequences = []
                nHandEmpty = 0
                aLastFrame = [-1, -1, -1, -1]

                # Analyze each frame -----------------------------------------------
                nIdxLoop = 0
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

                    if (sMode == 'flip') : image = cv2.flip(image, 1)

                    # Get features from image --------------------------------------
                    bCR, aHandLeft, aHandRight, aPoses, aFaces = mGetFeatures(image, sWord)
                    if not bCR : break

                    # Normalize feature : origin middle of shoulders, etc.  --------
                    aHandLeft, aHandRight, aPoses, aFaces = mNormalizeFeatures(aHandLeft, aHandRight, aPoses, aFaces)

                    # Aggregate pose, and hands features into a unique structure
                    # index 0: left hand, index 1: right hand, index 2: pose, index 3: face, 
                    aSequences.append((aHandLeft, aHandRight, aPoses, aFaces))

                    # Interpolate missing features ---------------------------------
                    for nIdxVector in aVectors :
                        # No interpolation for pose landmarks
                        if nIdxVector == cPOSE : continue

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

                # Suppress sequence features without hands -------------------------
                aSequencesCopy = []
                for aSequence in aSequences :
                    if (len(aSequence[cLEFT_HAND]) != 0) or (len(aSequence[cRIGHT_HAND]) != 0) :
                        aSequencesCopy.append(aSequence)
                aSequences = aSequencesCopy

                #print("... len(aSequences) : ", len(aSequences))
                nSequences = len(aSequences)

                # bde test - Number of empty landmarks <<<<<<<<<<<<<<<<<<<<<<<<<
                if (False) :
                    nHandTotal = 0
                    for aSequence in aSequences :
                        for nIdxVector in (cPOSE, cFACE) :
                            if len(aSequence[nIdxVector]) > 0 : nHandTotal += 1
                    if nHandTotal < 1 : nHandTotal = 1
                    print(f"... Hands missing;{sFile};{nHandEmpty};{nHandTotal};{round(nHandEmpty / nHandTotal, 3)}")
                    continue

                # bde test - Distance between 2 feature vectors (range 1..8) <<<
                if (False) :
                    nHorizonMax = 8
                    for nIdxFrame in range(nHorizonMax, nSequences) :
                        #aSequence = aSequences[nIdxFrame]
                        for nHorizon in range(1, min(nHorizonMax+1, nSequences)) :
                            aDistance = mGetDistance(aSequences[nIdxFrame], aSequences[nIdxFrame-nHorizon])
                            for nIdxVector in aVectors :
                                print(f"... distance;{nHorizon};{nIdxVector};{aDistance[nIdxVector]}")
                    continue

                # Before filtering and calculation, check that we have data to do that...
                if (nSequences < 1) :                

                    # After the feature extraction we having no data => it's an error !
                    print(">>> aSequences empty")

                else :

                    # Keep landmarks distance of minimal threshold...
                    aSamples, nSequences, nRowFeatures, nFeatures = mLsfFilterFrames(sFile, aSequences)
                    
                    # bde test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    #continue

                    aLstFeatures.append([sWord] + aSamples)

                    # --------------------------------------------------------------
                    # Feature augmentation : different scales for hands or pose, but the face doesn't change
                    # --------------------------------------------------------------
                    def mDataAugmented(aSamplesRef, nSign) :
                        #print(_getframe().f_code.co_name)

                        aSamples = copy.deepcopy(aSamplesRef)

                        # Hands scale : +/- 15% ------------------------------------
                        nHandSizeVar = 0.15
                        nHandSizeVar = 0.60
                        nHandScale = 1.0
                        while (abs(1.0 - nHandScale) < 0.0225) : nHandScale = 1.0 + nSign * nHandSizeVar * rd.random()
                        for nIdx in range(aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND]) :
                            aSamples[nIdx] = nHandScale * aSamplesRef[nIdx]

                        # Hands rotation : +/- 20° ---------------------------------
                        if ("rotation" in aDataAugmentation) :
                            nHandRotVar = 15
                            nHandRotVar = 60
                            nAngle = 0
                            while (abs(nAngle) < (2.25/180)) : nAngle = (2 * rd.random() - 1) * nHandRotVar * math.pi / 180
                            nCos = math.cos(nAngle)
                            nSin = math.sin(nAngle)
                            for nIdx in range(0, aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND], nDim) :
                                nLg = math.sqrt(aSamples[nIdx] * aSamples[nIdx] + aSamples[nIdx+1] * aSamples[nIdx+1] + aSamples[nIdx+2] * aSamples[nIdx+2])
                                aSamples[nIdx] = nLg * nCos
                                aSamples[nIdx+1] = nLg * nSin
                                # no change for z coordinate

                        # Pose scale : +/- 10% ------------------------------
                        if (len(aVectors) > cPOSE) : # check if pose (idx 2) is taking in count
                            nPoseSizeVar = 0.10
                            nPoseSizeVar = 0.40
                            nPoseScale = 1.0
                            while (abs(1.0 - nPoseScale) < 0.015) : nPoseScale = 1.0 + nSign * nPoseSizeVar * rd.random()
                            for nIdx in range(aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND], aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND] + aShapes[cPOSE]) :
                                aSamples[nIdx] = nPoseScale * aSamplesRef[nIdx]

                        # Pose rotation : +/- 20° ---------------------------
                        if ("rotation" in aDataAugmentation) :
                            # Arm rotation : +/- 20°
                            nPoseRotVar = 15
                            nPoseRotVar = 60
                            nAngle = 0
                            while (abs(nAngle) < (2.25/180)) : nAngle = (2 * rd.random() - 1) * nHandRotVar * math.pi / 180
                            nCos = math.cos(nAngle)
                            nSin = math.sin(nAngle)
                            for nIdx in range(aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND] + 3*nDim, aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND] + 7*nDim, nDim) :
                                nLg = math.sqrt(aSamples[nIdx] * aSamples[nIdx] + aSamples[nIdx+1] * aSamples[nIdx+1] + aSamples[nIdx+2] * aSamples[nIdx+2])
                                aSamples[nIdx] = nLg * nCos
                                aSamples[nIdx+1] = nLg * nSin

                        # Face scale : +/- 6% --------------------------------------
                        if (len(aVectors) > cFACE) : # check (idx cFACE) if face is taking in count
                            nFaceSizeVar = 0.06
                            nFaceSizeVar = 0.24
                            nFaceScale = 1.0
                            while (abs(1.0 - nFaceScale) < 0.009) : nFaceScale = 1.0 + nSign * nFaceSizeVar * rd.random()
                            for nIdx in range(aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND] + aShapes[cPOSE], aShapes[cLEFT_HAND] + aShapes[cRIGHT_HAND] + aShapes[cPOSE] + aShapes[cFACE]) :
                                aSamples[nIdx] = nFaceScale * aSamplesRef[nIdx]

                        return (aSamples)

                    # Data augmentation --------------------------------------------
                    if ("yes" in aDataAugmentation) :

                        aSamplesRef = copy.deepcopy(aSamples)

                        if ("scale" in aDataAugmentation) :
                            # Hands and pose larger or smaller
                            for nIdxDuplicate in range(nDuplicate) :
                                bLargerSmaller = True if (rd.random() > 0.5) else False
                                if bLargerSmaller :
                                    aSamples = mDataAugmented(aSamplesRef, 1.0)
                                    aLstFeatures.append([sWord] + aSamples)
                                else :
                                    aSamples = mDataAugmented(aSamplesRef, -1.0)
                                    aLstFeatures.append([sWord] + aSamples)

                # Clean everything before next loop --------------------------------
                if (sFileExt == ".mp4") :
                    camera.release()
                cv2.destroyAllWindows()

            if nFile > nFileLimit : break

    sDataFile = sDirModel + sDataFrame

    aColFeat = [str(i).zfill(nDim) for i in range(0, nSample * (aShapes[cLEFT_HAND]+aShapes[cRIGHT_HAND]+aShapes[cPOSE]+aShapes[cFACE]))]
    aCols = ["class"] + aColFeat

    df = pd.DataFrame(aLstFeatures, columns=aCols)
    df.to_csv(sDataFile, index=None)

    print("... delay : ", time.time() - nStartTime)

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
                sFilename = sDirModel + os.path.basename(sDataFrame) +'_'+ sDecompName +  '_decom.sav'
                joblib.dump(aReduct, sFilename) 

                #print("... explained_variance_       : ", sDecompName, aReductFit.explained_variance_)
                print("... explained_variance_ratio_ : ", sDecompName, sNormeName, sum(aReductFit.explained_variance_ratio_[:nComponents]))
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
def mLsfFeatureSelection(sDirModel, sDataFrame):
    #print(_getframe().f_code.co_name)

    # Analyse limited to 25% of features
    nFeatureVector = sum(aShapes) * nDim * nSample
    nFeature = nFeatureVector // 4

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
            print("sSelectName; sScoreName; aCols[nIdx]; nFrame; nIdxRel; nIdxFeat; sVector; sCoord; nLandMark; aScores[nIdx]")
            for nIdx in range(nFeature) :

                if nIdx >= 64 : break

                nIdxAbs = int(aCols[nIdx])
                # what frame
                nFrame = nIdxAbs // nFeatureVector

                # what part : left / right hand, pose, face
                nIdxRel = nIdxAbs % nFeatureVector
                if nIdxRel > nDim * (2*nFeatureHands + nFeaturePose) :
                    sVector = "face"
                    nIdxFeat = nIdxRel - nDim * (2*nFeatureHands + nFeaturePose)
                    nLandMark = aFacePoints[nIdxFeat//nDim]
                else :
                    if nIdxRel > nDim * (2*nFeatureHands) :
                        sVector = "pose"
                        nIdxFeat = nIdxRel - nDim * (2*nFeatureHands)
                        nLandMark = aPosePoints[nIdxFeat//nDim]
                    else :
                        if nIdxRel > nDim*(nFeatureHands) :
                            sVector = "right"
                            nIdxFeat = nIdxRel - nDim * (nFeatureHands)
                            nLandMark = aHandPoints[nIdxFeat//nDim]
                        else :
                            sVector = "left"
                            nIdxFeat = nIdxRel
                            nLandMark = aHandPoints[nIdxFeat//nDim]
                sCoord = aCoord[nIdxFeat % nDim]          

                print(f'{sSelectName};{sScoreName};{aCols[nIdx]};{nFrame};{nIdxRel};{nIdxFeat};{sVector};{sCoord};{nLandMark};{aScores[nIdx]}')

    return

# ------------------------------------------------------------------------------
# Train and test with different algorithms (classifiers and normalizers)
# ------------------------------------------------------------------------------
def mLsfTrainTest(sDirModel, sDataFrame):
    #print(_getframe().f_code.co_name)

    # Get data normalize
    sPathDataFrame = sDirModel + sDataFrame
    aXInits, aYInits = mLsfLoadDataSet(sPathDataFrame)

    # Normalizer ---------------------------------------------------------------
    for sNormeName, oScaler in aNormalizers.items():

        aXs = aXInits.copy(deep=True)
        aYs = aYInits.copy(deep=True)

        # Normalize values -----------------------------------------------------
        aXs = oScaler.fit_transform(aXs.values)

        # Save rule to future normalyze input data
        sFilename = sDirModel + os.path.basename(sDataFrame) + '_' + sNormeName + '_norm.sav'
        joblib.dump(oScaler, sFilename) 

        # Create train and test set --------------------------------------------
        sYBefore = "---"

        aXTrain = []
        aYTrain = []
        aXTest = []
        aYTest = []

        bFirst = True      # 1st occurrence used as data test, or second
        for aX, sY in zip(aXs, aYs) :

            # Algo 1 : select N first examples / sign --------------------------
            if True :
                if (sYBefore != sY) : nNew = 0

                if nNew < 6 : # bde test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    aXTest.append(aX)
                    aYTest.append(sY)
                else :
                    aXTrain.append(aX)
                    aYTrain.append(sY)

                nNew += 1
                sYBefore = sY

            # Algo 2 : select X% examples / sign for test ----------------------
            if False :
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
            sFilename = sDirModel + os.path.basename(sDataFrame) +'_'+ sClassifierName + '_' + sNormeName +  '_clas.sav'
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
            sFilename = sDirModel + os.path.basename(sDataFrame) +'_'+ sClassifierName + '_' + sNormeName +  '_clas.sav'
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
# Test the models with real time video capture
# ------------------------------------------------------------------------------
aResultSpeech = ("Oups, incorrect", "Pas mal, peut mieux faire", "Bien, encore un effort", "Parfait, bravo")
aPrattKernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])

def mLsfInference(sDirModel, sDataFrame, mp_hands, mp_pose, mp_faceMesh, hands, pose, faceMesh):
    #print(_getframe().f_code.co_name)

    global sWordDemo, bNextVideo

    # Initilize video capture
    oCamera, nWidth, nHeight = mLsfInitCamera()
    sFile = "camera"

    sWord = ""
    nAttemptMax = 3
    nAttempt = 0
    bNextAttempt = True
    while oCamera.isOpened():

        nAttempt += 1

        #if sWord == 'au_revoir' : break
        if bNextAttempt :
            print()
            print(f"... sign : {sWordDemo} ... {nAttempt} / {nAttemptMax}")
        bNextAttempt = False

        aSequences = []
        aLastFrame = [-1, -1, -1, -1]
        nHandEmpty = 0

        # Get a word...
        nIdxLoop = 0
        while oCamera.isOpened():

            bRetCode, image = oCamera.read()
            if not bRetCode : break
            image = cv2.resize(image, (nWidth, nHeight))
            
            """ test contrast
            # Apply Paratt kernel to the image
            #image = cv2.filter2D(image, -1, aPrattKernel)
            #image = cv2.equalizeHist(image)

            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
            l, a, b = cv2.split(lab)  # split on 3 different channels
            l2 = clahe.apply(l)  # apply CLAHE to the L-channel
            lab = cv2.merge((l2,a,b))  # merge channels
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
            """

            # Get features from image ------------------------------------------
            bCR, aHandLeft, aHandRight, aPoses, aFaces = mGetFeatures(image, "Test your knowledge (hide your hands between the signs)...")
            if not bCR : break

            # Normalize feature : origin middle of shoulders, etc.  ------------
            aHandLeft, aHandRight, aPoses, aFaces = mNormalizeFeatures(aHandLeft, aHandRight, aPoses, aFaces)

            # Aggregate pose, and hands features into a unique structure
            # index 0: left hand, index 1: right hand, index 2: pose, index 3: face, 
            aSequences.append((aHandLeft, aHandRight, aPoses, aFaces))

            # Interpolate missing features -------------------------------------
            for nIdxVector in aVectors :
                # No interpolation for pose landmarks
                if nIdxVector == cPOSE : continue

                nLength = len(aSequences[-1][nIdxVector])
                if nLength > 0 :
                    nLast = aLastFrame[nIdxVector]
                    if nLast < 0 :
                        nLast = nIdxLoop

                    elif nLast < (nIdxLoop - 1) :
                        # Interpolation needed
                        # ATTENTION : compteur nHandEmpty compte aussi les interpolations de "face"
                        nHandEmpty += mInterpolatePart(aLandmarkRef[nIdxVector], nIdxVector, nLast, nLength, nIdxLoop, aSequences)

                    aLastFrame[nIdxVector] = nIdxLoop

            nIdxLoop += 1

            # Criteria to start recording sign ---------------------------------
            if aLastFrame[cLEFT_HAND] == -1 and aLastFrame[cRIGHT_HAND] == -1 :
                nIdxLoop = 0
                aSequences = []
                aLastFrame = [-1, -1, -1, -1]
                continue

            # Criteria for end sign --------------------------------------------
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
            nViewHand = ((aLastFrame[cLEFT_HAND] + aLastFrame[cRIGHT_HAND]) > -2)
            if bNoHands and bBigHole and nViewHand : break

        # Suppress sequence features without hands -------------------------
        aSequencesCopy = []
        for aSequence in aSequences :
            if (len(aSequence[cLEFT_HAND]) != 0) or (len(aSequence[cRIGHT_HAND]) != 0) :
                aSequencesCopy.append(aSequence)
        aSequences = aSequencesCopy
        
        # Before filtering and calculation, check that we have data to do that...
        #print("... len(aSequences) : ", len(aSequences))
        nSequences = len(aSequences)
        if (nSequences > 1) :
            # Keep landmarks distance of minimal threshold...
            aSamples, nSequences, nRowFeatures, nFeatures = mLsfFilterFrames(sFile, aSequences)
        else :
            # Not anougth data...
            #print(".. Not anougth data... : ", nSequences)
            nAttempt -= 1
            continue

        # Check dimension before continuing
        #print("... len(aSamples) ? nSample*sum(aShapes) : ", len(aSamples), nSample * sum(aShapes))
        if (len(aSamples) != (nSample * sum(aShapes))) : continue

        # Different data group size
        nFeatureHands = 2 * nDim * len(aHandPoints)
        nFeaturePose = nDim * len(aPosePoints)
        nFeatureFace = nDim * len(aFacePoints)
        nFeatureAll = nFeatureHands + nFeaturePose + nFeatureFace

        # Keep only data group selected ----------------------------------------
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

            sFilename = sDirModel + os.path.basename(sDataFrame) + '_' + sNormeName + '_norm.sav'
            oScaler = joblib.load(sFilename)
            aSamples = oScaler.transform([aSamples])

            # Test -------------------------------------------------------------
            for sClassifierName, oclassifier in aClassifiers.items():

                sFilename = sDirModel + os.path.basename(sDataFrame) +'_'+ sClassifierName + '_' + sNormeName +  '_clas.sav'
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

                # bde test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                if False : print(f"... {sClassifierName:8} + {sNormeName:8} => {sWord:16} à {round(100*nPredict, 1)}")

        # Select most frequence proposition ------------------------------------
        nSize = len(aWord)
        sWord = max(set(aWord), key = aWord.count)

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
            aResult.append((sWordL, round(100*nProba/nSize,1), nWord))
            #print(f"...... {sWordL:16} {nWord}x {round(100*nProba/nSize,1):3.1f}%")

        #aResult = sorted(aResult, key=operator.itemgetter(2, 1), reverse=True)
        aResult = sorted(aResult, key=operator.itemgetter(1, 2), reverse=True)

        nPoint = 0
        print("... detected :")
        for sWordL, nProba, nWord in aResult :
            print(f"...... {sWordL:24} {nProba:3.1f}% {nWord}x")
            if (sWordDemo == sWordL) : nPoint = 1
        
        if (sWordDemo == aResult[0][0]) :
            nPoint = 3 if ((len(aResult) == 1) or (aResult[0][1] > 50.)) else 2
        """
        if (sAction == "demo") :
            print("... point : ", nPoint)
            try :
                speak(aResultSpeech[nPoint], "fr", save=False, file="point.mp3", speak=True)
            except :
                pass
        """
        
        # Get N chances before changing sign / word ----------------------------
        if (nAttempt >= nAttemptMax) :
            bNextVideo = True
            nAttempt = 0
            time.sleep(2)

        sWordDemo = sWordDemoNew
        bNextAttempt = True

    cv2.destroyAllWindows()

# ------------------------------------------------------------------------------
# Play and example of video sign...
# ------------------------------------------------------------------------------
def mLstGetExample(sDirIn, sSrc, sChoice) :
    #print(_getframe().f_code.co_name)
    
    global sWordDemoNew, bNextVideo
    
    nWaitTime = 40
    nSlowRate = 2
    nVideoMax = 4
    
    nFont = cv2.FONT_HERSHEY_SIMPLEX
    nFontScale = 2
    nThickness = 3
    nLineType = cv2.LINE_AA

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

        nIdxFile = -1
        nFileMax = len(aFiles)

        print("... len() {} / {} ".format(nFileLimit, nFileMax))

        while (True) :

            # Rule to select input file ----------------------------------------
            if (sChoice == "alea") : 
                #nIdxFile = int(nFileMax * rd.random())
                nIdxFile = rd.randint(0, nFileMax)
            else :
                nIdxFile += 20
                #if (nIdxFile >= nFileMax) : break
                if (nIdxFile >= nFileMax) : nIdxFile = 0

            sFile = aFiles[nIdxFile]
            sFileExt = sFile[-4:]
            if (sFileExt != ".mp4") : 
                continue

            sBaseFile = os.path.basename(sFile)
            sWordDemoNew = sBaseFile.split("#")[0]
            #print(f"... next word : {nIdxFile} / {nFileMax} => {sWordDemoNew} {sBaseFile}")

            # Play the video several time --------------------------------------
            nIdxVideo = 0
            #while (nIdxVideo < nVideoMax) :
            bNextVideo = False
            while (not bNextVideo) :

                # Open stream on the file --------------------------------------
                camera = cv2.VideoCapture(sDirSrc + sFile)

                while (True) :
                
                    if (sFileExt == ".mp4") :
                        success, image = camera.read()
                        if not success : break
                    image = cv2.resize(image, (nWidth, nHeight))

                    cv2.putText(image, sWordDemoNew, (0, 45), nFont, nFontScale, (0, 0, 255), nThickness, nLineType)

                    for nIdxSlowRate in range(nSlowRate) :
                        cv2.imshow(sWinName, image)
                        time.sleep(0.040)
                    
                    cv2.waitKey(nWaitTime)
                time.sleep(0.5)
                nIdxVideo += 1

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
def mLsfAddVideo(mp_hands, mp_pose, mp_faceMesh, hands, pose, faceMesh, sDirIn, sSrc) :
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
                bCR, aHandLeft, aHandRight, aPoses, aFaces = mGetFeatures(image, "image")
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
                nViewHand = ((aLastFrame[cLEFT_HAND] + aLastFrame[cRIGHT_HAND]) > -2)
                if bNoHands and bBigHole and nViewHand : break

                # bde test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                if False :
                    sFilenamePicture = "{}#A#{}#{}#fablab#{}_{}_{}.jpg".format(sWord.replace(' ', '_'), nIdx, sNature, sKey, sDate, nFrame)
                    sTmpPicture = './tmp/' + sFilenamePicture
                    cv2.imwrite(sTmpPicture, image)

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

    nArg = len(sys.argv)
    
    # Get screen size
    #nWidthScreen, nHeightScreen = mLsfGetScreenSize()
    nWidthScreen = 1508

    # Get user's parameters or default values
    #---------------------------------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--in",   default="./select/", help="input files directory")
    ap.add_argument("-m", "--mod",  default="./model/", help="output / input dataframe / model directory")
    ap.add_argument("-s", "--src",  default="xxx", help="video source (dicoelix, pisourd, sematos, other, xxx, test)")
    ap.add_argument("-e", "--set",  default="directory", help="video set : directory, array, test")
    ap.add_argument("-n", "--nb",   default="0", help="how many video, 0 = all")
    ap.add_argument("-c", "--cho",  default="sequence", help="video choice : alea, sequence")
    ap.add_argument("-t", "--dis",  default="hands", help="display video on screen : hands, pose, face, yes, no")
    ap.add_argument("-d", "--df",   default="dicolsf_fl_xxx_3_8S.npy", help="dataframe name")
    ap.add_argument("-a", "--action", default="infe", help="vide[o], feat[ure], redu[ction], sele[ction], trai[n&test], infe[rence], demo[nstration]")
    ap.add_argument("-g", "--aug",  default="no", help="data augmentation : yes, no, flip, scale, rotation")
    ap.add_argument("-x", "--naug", default="1", help="data augmentation : how many duplicate values")
    ap.add_argument("-v", "--vec",  default="hands,pose", help="vector features composed of: hands[,pose[,face]]")
    ap.add_argument("-z", "--dim",  default="3", help="vector features dimension 2 (x,y) or 3 (x,y,z)")
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
        #if s1Src not in ('dicoelix', 'pisourd', 'sematos', 'other', 'fablab', 'alpha', 'xxx', 'xxx-bde', 'test', 'testspeed') : 
            bParamError = True
            print(">>> src    : error value")
    sSet = args['set'].lower()
    if sSet not in ('directory', 'array', 'test') : 
        bParamError = True
        print(">>> set    : error value")
    nFileLimit = int(args['nb'])
    if (nFileLimit < 1) : nFileLimit = 9999999999
    sChoice = args['cho'].lower()
    if sChoice not in ('alea', 'sequence') : 
        bParamError = True
        print(">>> cho    : error value")

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
    #nWaitTime = 5

    sDataFrame = args['df']
    sAction = args['action'].lower()[0:4]
    if sAction not in ('vide', 'feat', 'redu', 'sele', 'trai', 'infe', 'demo') : 
        bParamError = True
        print(">>> action : error value")

    sDataAugmentation = args['aug'].lower()
    aDataAugmentation = sDataAugmentation.split(',')
    if ("yes" in aDataAugmentation) :
        aDataAugmentation.append("flip")
        aDataAugmentation.append("rotation")
        aDataAugmentation.append("scale")
    if ("flip" in aDataAugmentation) or ("scale" in aDataAugmentation) or ("rotation" in aDataAugmentation) :
        aDataAugmentation.append("yes")
    nDuplicate = int(args['naug'])

    # Which features : hands[,pose[,face]]
    # Threshold between 2 frames, which we estimate as significant for detecting motion (Cf. xlsx / dist feat.)
    nDim = int(args['dim'])
    if nDim == 2 :
        aDistThreshold = [0.0850, 0.0850, 0.110, 0.0850]
        aDistThreshold = [0.0630, 0.0630, 0.0768, 0.0598]
    else :
        aDistThreshold = [0.170, 0.170, 0.470, 0.155] # => avg 9 frames/sign
        aDistThreshold = [0.207, 0.207, 0.545, 0.155] # => avg 8 frames/sign = 10.1 si h|p
        aDistThreshold = [0.255, 0.255, 0.665, 0.155] # => avg 7 frames/sign =  9.0 si h|p
        aDistThreshold = [0.325, 0.325, 0.845, 0.155] # => avg 6 frames/sign =  7.8 si h|p
        aDistThreshold = [0.550, 0.550, 1.250, 0.155] # => avg 4 frames/sign =  5.8 si h|P

        aDistThreshold = [0.207, 0.207, 0.545, 0.155]
    nSample = 8                 # number frame by word
        
    sVectorFeatures = args['vec']
    aVectorFeatures = sVectorFeatures.split(',')
    aVectors = [cLEFT_HAND, cRIGHT_HAND, cPOSE, cFACE]
    aShapes = [nDim*len(aHandPoints), nDim*len(aHandPoints), nDim*len(aPosePoints), nDim*len(aFacePoints)]

    sInterPolation = "wrist"     # wrist, linear

    if (nArg == 1) or bParamError :
        print("usage : ")
        print("      --in     default='./select/', input files directory")
        print("      --mod    default='./model/', output / input dataframe / model directory")
        print("      --src    default='xxx', video source (dicoelix, pisourd, sematos, other, xxx, test)")
        print("      --set    default='directory', video set : directory, array, test")
        print("      --nb     default='0', how many video, 0 = all")
        print("      --cho    default='sequence', video choice : alea, sequence")
        print("      --dis    default='hands', display video on screen : hands, pose, face, yes, no")
        print("      --df     default='dicolsf_fl_xxx_3_8S.npy', dataframe name")
        print("      --action default='infe', what action to do : vide[o], feat[ure], redu[ction], sele[ction], trai[n&test], infe[rence], demo[nstration]")
        print("      --aug    default='no', data augmentation : flip, scale, rotation, yes, no")
        print("      --naug   default='1', data augmentation : how many duplicate values")
        print("      --vec    default='hands,pose', vector features composed of: hands[,pose[,face]]")
        print("      --dim    default='3', vector features dimension 2 (x,y) or 3 (x,y,z)")
        print("... at least one argument needed")
        print()
        print(f"... python {sys.argv[0]} --in {sDirIn} --mod {sDirModel} --set {sSet} --nb {nFileLimit} --cho {sChoice} --aug {sDataAugmentation} --naug {nDuplicate} --dis {sDisplay} --vec {sVectorFeatures} --dim {nDim} --src {sSrc} --df {sDataFrame} --action {sAction}")
        print()

        exit(1)

    print(f"... python {sys.argv[0]} --in {sDirIn} --mod {sDirModel} --set {sSet} --nb {nFileLimit} --cho {sChoice} --aug {sDataAugmentation} --naug {nDuplicate} --dis {sDisplay} --vec {sVectorFeatures} --dim {nDim} --src {sSrc} --df {sDataFrame} --action {sAction}")

    # Initialize MediaPipe package ---------------------------------------------
    mp_drawing, mp_drawing_styles, mp_hands, mp_pose, mp_faceMesh, hands, pose, faceMesh = mLsfInitMediaPipe()

    if (sAction == "vide") :
        ans = mLsfAddVideo(mp_hands, mp_pose, mp_faceMesh, hands, pose, faceMesh, sDirIn, sSrc)

    if (sAction == "feat") :
        ans = mLsfFeature(sDirIn, sDirModel, sDataFrame, mp_hands, mp_pose, mp_faceMesh, hands, pose, faceMesh)

    if (sAction == "redu") :
        ans = mLsfReduction(sDirModel, sDataFrame)

    if (sAction == "sele") :
        ans = mLsfFeatureSelection(sDirModel, sDataFrame)

    if (sAction == "trai") :
        ans = mLsfTrainTest(sDirModel, sDataFrame)

    if (sAction == "infe") :
        ans = mLsfInference(sDirModel, sDataFrame, mp_hands, mp_pose, mp_faceMesh, hands, pose, faceMesh)

    if (sAction == "demo") :

        # Start the thread showing video examples
        oThreadVideo = threading.Thread(target=mLstGetExample, args=(sDirIn, sSrc, sChoice))
        oThreadVideo.daemon = True
        oThreadVideo.start()

        # Main code used to verify your attempts
        ans = mLsfInference(sDirModel, sDataFrame, mp_hands, mp_pose, mp_faceMesh, hands, pose, faceMesh)
