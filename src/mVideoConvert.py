# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

# import the necessary packages ------------------------------------------------

from typing import NamedTuple

import argparse
import json
import os
import subprocess
import sys
import time

# Define global variables ------------------------------------------------------

sDirIn = './in/'
sDirOut = './out/'

aExtensionVideo = ('asf', 'avi', 'mp4', 'm4v', 'mov', 'mpg', 'mpeg', 'wmv', 'qt', 'flv', 'vob', 'mkv')

class FFProbeResult(NamedTuple):
    return_code: int
    json: str
    error: str


def ffprobe(file_path) -> FFProbeResult:
    command_array = ["ffprobe",
                     "-v", "quiet",
                     "-print_format", "json",
                     "-show_format",
                     "-show_streams",
                     file_path]
    result = subprocess.run(command_array, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return FFProbeResult(return_code=result.returncode,
                         json=result.stdout,
                         error=result.stderr)

def mFileConvert(sFileIn, sSize, sDirOut = './out/', bVerbose = True) :

    nStartTime = time.time()

    if (sSize in ("S", "VGA", "640")) : nTargetSize = 640
    if (sSize in ("DVD", "720")) : nTargetSize = 720
    if (sSize in ("SVGA", "800")) : nTargetSize = 800
    if (sSize in ("XGA", "1024")) : nTargetSize = 1024
    if (sSize in ("M", "SXGA", "HDR", "1280")) : nTargetSize = 1280
    if (sSize in ("WXGA", "1366")) : nTargetSize = 1366
    if (sSize in ("UXGA", "1600")) : nTargetSize = 1600
    if (sSize in ("L", "FHD", "1920")) : nTargetSize = 1920
    if (sSize in ("2K", "1998")) : nTargetSize = 1998
    if (sSize in ("QHD", "2560")) : nTargetSize = 2560
    if (sSize in ("4K", "3840")) : nTargetSize = 3840
    if (sSize in ("8K", "7680")) : nTargetSize = 7680

    sFileBasename = os.path.basename(sFileIn)
    aFile = os.path.splitext(sFileBasename)
    sFileName = aFile[0]
    sFileExt = aFile[1][1:]

    # Check video file extension -----------------------------------------------
    if (sFileExt.lower() not in aExtensionVideo) : return 0

    sFileOut = sDirOut + sFileName.replace(" ", "_") + ".mp4"
    
    print(sFileIn, sFileOut, sSize)

    # Get video size -----------------------------------------------------------
    try :
        ffprobe_result = ffprobe(file_path=sFileIn)
        if (ffprobe_result.return_code != 0) :
            print(">>> ERROR", ffprobe_result.error, file=sys.stderr)
            return 0

        # Print the raw json string
        sResult = ffprobe_result.json
        #print(sResult)
        aJsonResult = json.loads(sResult)
        #print(aJsonResult['streams'][0])
        #print(aJsonResult['streams'][1])

        # Calculate target size depend on user ask ---------------------------------
        nWidth = int(aJsonResult['streams'][0]['width'])
        nHeight = int(aJsonResult['streams'][0]['height'])
        nDuration = float(aJsonResult['streams'][0]['duration'])
    except :
        print(">>> ffprobe error <{}>".format(sFileIn))
        return 0

    """
    # Special format
    nWidth = 360
    nHeight = 640
    """
    
    nCoeff = round(nWidth / 1000.0 * nHeight / 1000.0 * nDuration, 2)
    nConvDuration = round(0.4925 * nCoeff + 10.51, 0)
    if (nConvDuration < 1) : nConvDuration = 1

    # Keep size ratio between sides
    if (nWidth >= nHeight) :
        nTargetWidth = nTargetSize
        nTargetHeight = 2 * int(round((nHeight / nWidth * nTargetSize) / 2, 0))
    else :
        nTargetHeight = nTargetSize
        nTargetWidth = 2 * int(round((nWidth / nHeight * nTargetSize) / 2, 0))

    if bVerbose : print("... from <{}> {}x{} to {}x{} <{}>, estimation duration {} s".format(sFileIn, nWidth, nHeight, nTargetWidth, nTargetHeight, sFileOut, nConvDuration))

    # Use ffmpeg to convert your video file ------------------------------------
    nFPS = 25
    try :
        sCmd = ["ffmpeg",
                 "-r", str(nFPS),
                 "-i", sFileIn,
                 "-vf", "scale="+str(nTargetWidth)+":"+str(nTargetHeight),
                 "-ar", "22050",
                 "-b:a", "44100",
                 "-crf", "28",
                 "-y",
                 sFileOut]
        oProbe = subprocess.Popen(sCmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        """
        sCR = oProbe.wait()
        print("sCR : ", sCR)
        """
        sOut, sErr =  oProbe.communicate()
    except :
        print(">>> ffmpeg error <{}>".format(sFileIn))
        sOut = "error"
        sErr = "error"

    #print("... => {};{}".format(nCoeff, round(time.time() - nStartTime, 0)))
    """
    print("... sOut : ", sOut)
    print("... sErr : ", sErr)
    """

    return 1

# ------------------------------------------------------------------------------
# mCheckUserParameters() 
# ------------------------------------------------------------------------------
def mCheckUserParameters() :
    #print(_getframe().f_code.co_name)

    # Check  parameters
    #---------------------------------------------------------------------------
    bParameterError = False

    if (not os.path.isdir(sDirIn)) :
        print(">>> --dirin <{0}> must be a directory".format(sDirIn))
        bParameterError = True

    if (not os.path.isdir(sDirOut)) :
        print(">>> --dirin <{0}> must be a directory".format(sDirOut))
        bParameterError = True

    if (sSize not in ('2K', '4K', '8K', 'DVD', 'FHD', 'HDR', 'L', 'M', 'QHD', 'S', 'SVGA', 'SXGA', 'UXGA', 'VGA', 'WXGA', 'XGA')) :
        print(">>> --get <{0}> must be in (2K, 4K, 8K, DVD, FHD, HDR, L, M, QHD, S, SVGA, SXGA, UXGA, VGA, WXGA, XGA)".format(sSize))
        bParameterError = True

    return bParameterError

# ------------------------------------------------------------------------------
# main() 
# ------------------------------------------------------------------------------
if __name__ == '__main__' :
    #print("__name__ : ", __name__)

    # Get user's parameters or default values
    #---------------------------------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--in", default="./in/", help="input files directory")
    ap.add_argument("-o", "--out", default="./out/", help="output files directory")
    ap.add_argument("-s", "--size", default="M", help="video size : (L)arge 1920x1080, (M)edium 1280x720, (S)mall 640x360")
    args = vars(ap.parse_args())

    sDirIn = args['in']
    sDirOut = args['out']
    sSize = args['size'].upper()

    # Check users's parameters -------------------------------------------------
    bParameterError = mCheckUserParameters()

    print("[INFO] How to use this script...")
    print("... python mVideoConvert.py --input {} --output {} --size {}".format(sDirIn, sDirOut, sSize))
    print("        --input  : input video files directory [default ./in/]")
    print("        --output : output video files directory [default /out/]")
    print("        --size   : video size, [default M]")
    print("                   S = 640x360, VGA = 640x480, DVD = 720x480, SVGA = 800x600, XGA = 1024x768, HDR = 1280x720,")
    print("                   M = 1280x720, SXGA = 1280x1024, WXGA = 1366x768, UXGA = 1600x1200, FHD = 1920x1080,")
    print("                   L = 1920x1080, 2K = 1998x1080, QHD = 2560x1440, 4K = 3840x2160, 8K = 7680x4320,")
    print()

    # Exit in case of parameters error or help parameter
    if (bParameterError) : sys.exit(1)

    # For all files in input directory -----------------------------------------
    for sFile in os.listdir(sDirIn) :

        # path input file
        sFileIn = sDirIn + sFile
        
        mFileConvert(sFileIn, sSize, sDirOut = sDirOut)
