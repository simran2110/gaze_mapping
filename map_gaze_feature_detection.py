""" Map gaze data from world camera coordinate system to reference image

This script automates mapping of gaze data from a world camera coordinate
system to a fixed reference image. Mobile eye-trackers often record gaze data
with respect to an outward facing world camera approximating the
participant's point-of-view. As a result, the gaze data is expressed in
an egocentric coordinate system which moves along with the participant's head.

Typical eye-tracking research, on the other hand, seeks to analyze gaze
behavior on a particular stimulus over time. In order to use mobile
eye-trackers in this context, one must first map the recorded gaze points from
the world camera coordinate system to the fixed coordinate system of the
target stimulus. This requires 1) identifying the target stimulus in every
frame of the world camera recording, 2) finding a linear transform that will
map between the appearance of the stimulus on the world camera frame and a 2D
reference version of the same stimulus, and 3) using that transform to project
the recorded gaze points to the 2D reference stimulus.

With the help of computers vision tools, this script automates this process
and yeilds output data files that facilitate subsequent analysis, specifically:
    - world_gaze.m4v:           world video w/ gaze points overlaid
    - ref_gaze.m4v:             video of ref image w/ gaze points overlaid
    - ref2world_mapping.m4v     video of reference image projected back into
                                world video
    - gazeData_mapped.tsv:      gazeData mapped to both coordinate systems, the
                                world and reference image

"""

from __future__ import division
from __future__ import print_function

import os
import sys
from os.path import join
import logging
import shutil
import time
import argparse

import numpy as np
import pandas as pd
import cv2

global OPENCV3
OPENCV3 = (cv2.__version__.split('.')[0] == '3')
print("OPENCV version " + cv2.__version__)


def findMatches(img1_kp, img1_des, img2_kp, img2_des):
    """ Find the matches between the descriptors for two images

    Parameters
    ----------
    img1_kp, img2_kp : list
        list of identified keypoints for each image; returned from
        detectAndCompute method on the cv2 featureDetect class.
    img1_des, img2_des : np.ndarray
        descriptors for each image; returned from detectAndCompute method on
        the cv2 featureDetect class.

    Returns
    -------
    img1_pts, img2_pts : list or None
        list of matched keypoints on each image

    """
    # Match settings
    min_good_matches = 4
    num_matches = 2
    FLANN_INDEX_KDTREE = 0
    distance_ratio = 0.5                # 0-1; lower values more conservative
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=10)        # lower = faster, less accurate
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    # find all matches
    matches = matcher.knnMatch(img1_des, img2_des, k=num_matches)

    # filter out cases where the 2 matches are too close to each other
    goodMatches = []
    for m, n in matches:
        if m.distance < distance_ratio * n.distance:
            goodMatches.append(m)

    if len(goodMatches) > min_good_matches:
        img1_pts = np.float32([img1_kp[i.queryIdx].pt for i in goodMatches])
        img2_pts = np.float32([img2_kp[i.trainIdx].pt for i in goodMatches])

        return img1_pts, img2_pts

    else:
        return None, None


def mapCoords2D(coords, transform2D):
    """ Map the supplied coords to a new coordinate system using the supplied
    transformation matrix

    Parameters
    ----------
    coords : tuple
        (x,y) coordinates
    transform2D : np.ndarray
        2D transformation matrix; produce by cv2.findHomography

    Returns
    -------
    float, float
        mapped coordinates after applying transform2D

    """

    coords = np.array(coords).reshape(-1, 1, 2)
    mappedCoords = cv2.perspectiveTransform(coords, transform2D)
    mappedCoords = np.round(mappedCoords.ravel())

    return mappedCoords[0], mappedCoords[1]


def projectImage2D(origFrame, transform2D, newImage):
    """ Project newImage into the origFrame

    Warp newImage according to the supplied transformation matrix, then
    project (insert) into the original frame.

    Parameters
    ----------
    origFrame : np.ndarray
        The original image you want to insert the newImage into
    transform2D : np.ndarray
        2D transformation matrix; produce by cv2.findHomography
    newImage : np.ndarray
        The image you would like to warp and project into the origFrame


    Returns
    -------
    newFrame : np.ndarray
        New frame (same dimensions as origFrame) with the warped and projected
        newImage written into it

    """
    # warp the new image to the video frame
    warpedImage = cv2.warpPerspective(newImage,
                                      transform2D,
                                      origFrame.T.shape[1:])

    # mask and subtract new image from video frame
    warpedImage_bw = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2GRAY)
    if warpedImage.shape[2] == 4:
        alpha = warpedImage[:, :, 3]
        alpha[alpha == 255] = 1       # create mask of non-transparent pixels
        warpedImage_bw = cv2.multiply(warpedImage_bw, alpha)

    ret, mask = cv2.threshold(warpedImage_bw, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    origFrame_bg = cv2.bitwise_and(origFrame, origFrame, mask=mask_inv)

    # mask the warped new image, and add to the masked background frame
    warpedImage_fg = cv2.bitwise_and(warpedImage[:, :, :3],
                                     warpedImage[:, :, :3],
                                     mask=mask)
    newFrame = cv2.add(origFrame_bg, warpedImage_fg)

    # return the warped new frame
    return newFrame


def processRecording(gazeData=None, worldCameraVid=None, referenceImage=None, outputDir=None, nFrames=None):
    """ Map the gaze across all frames of mobile eye-tracking session

    This method will iterate over every frame of the supplied video recording.
    On each frame, it will look for the matches with the specified
    referenceImage, create a linear transformation matrix, and map the gaze
    data from the world camera coordinate system to the reference image
    coordinate system.

    This parent method will take care of setting up all of the inputs, and at
    the end, writing all of the output files

    Parameters
    ----------
    gazeData : string
        Path to the gazeData file. This file expected to be a .csv/.tsv file
        with columns for:
            timestamp - timestamp (ms) corresponding to each sample
            frame_idx - index (0-based) of the worldCameraVid frame
                        corresponding to each sample
            confidence - confidence of the validity of each sample (0-1)
            norm_pos_x - normalized x position of gaze location (0-1).
                         Normalized with respect to width of worldCameraVid
            norm_pos_y - normalized y position of gaze location (0-1).
                         Normalized with respect to height of worldCameraVid
    worldCameraVid : string
        Path to the video recording from the world camera (.mp4)
    referenceImage : string
        Path to the 2D reference image
    outputDir : string
        Path to output directory where data will be saved
    nFrames : int, optional
        If specified, will only process given number of frames (default of
        None means it will process ALL frames in the video). Useful for testing
        on abbreviated number of frames

    Output files
    ------------
    world_gaze.m4v : video
        world video with original gaze points overlaid
    ref_gaze.m4v : video
         ref image with mapped gaze points overlaid
    ref2world_mapping.m4v : video
        world video with reference image projected and inserted into it.
    gazeData_mapped.tsv :  data file
        gazeData represented in both coordinate systems, the world and
        reference image

    """
    # Create output directory
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)

    # Set up Logging
    fileLogger = logging.FileHandler(join(outputDir, 'mapGazeLog.log'), mode='w')
    fileLogger.setLevel(logging.DEBUG)
    fileLogFormat = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%m-%d %H:%M:%S')
    fileLogger.setFormatter(fileLogFormat)
    consoleLogger = logging.StreamHandler(sys.stdout)
    consoleLogger.setLevel(logging.INFO)
    consoleLogFormat = logging.Formatter('%(message)s')
    consoleLogger.setFormatter(consoleLogFormat)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fileLogger)
    logger.addHandler(consoleLogger)

    # Log Inputs
    logger.info('Gaze Data File: {}'.format(gazeData))
    logger.info('World Camera Video: {}'.format(worldCameraVid))
    logger.info('Reference Image: {}'.format(referenceImage))
    logger.info('Output Directory: {}'.format(outputDir))

    # Copy the reference stim into the output dir
    shutil.copy(referenceImage, outputDir)

    # Load gaze data
    gazeWorld_df = pd.read_table(gazeData, sep='\t')
    print(gazeWorld_df)
    
    # gazeWorld_df = pd.read_csv(gazeData, sep=",")

    # Load the reference image
    refImg = cv2.imread(join(outputDir, referenceImage.split('/')[-1]))
    refImgColor = refImg.copy()      # store a color copy of the image
    refImg = cv2.cvtColor(refImg, cv2.COLOR_BGR2GRAY)  # convert the orig to bw

    ### Prep the video data #######################################
    # Load the video, get parameters
    vid = cv2.VideoCapture(worldCameraVid)
    if cv2.__version__.split('.')[0] >= '3':
        totalFrames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        vidSize = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = vid.get(cv2.CAP_PROP_FPS)
        vidCodec = cv2.VideoWriter_fourcc(*'mp4v')
        featureDetect = cv2.SIFT_create()
    else:
        totalFrames = vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        vidSize = (int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
        fps = vid.get(cv2.cv.CV_CAP_PROP_FPS)
        vidCodec = cv2.cv.CV_FOURCC(*'mp4v')
        featureDetect = cv2.SIFT()
    print("Total frames in the video are ",totalFrames)
    print("Video resolution ",vidSize)
    
    
    # World camera output video
    vidOut_world_fname = join(outputDir, 'world_gaze.m4v')
    vidOut_world = cv2.VideoWriter()
    vidOut_world.open(vidOut_world_fname, vidCodec, fps, vidSize, True)

    # Reference image output video
    vidOut_ref_fname = join(outputDir, 'ref_gaze.m4v')
    vidOut_ref = cv2.VideoWriter()
    vidOut_ref.open(vidOut_ref_fname,
                    vidCodec,
                    fps,
                    (refImg.shape[1], refImg.shape[0]),
                    True)

    # Ref2world mapping output video (useful for debugging)
    vidOut_ref2world_fname = join(outputDir, 'ref2world_mapping.m4v')
    vidOut_ref2world = cv2.VideoWriter()
    vidOut_ref2world.open(vidOut_ref2world_fname, vidCodec, fps, vidSize, True)

    ### Find keypoints, descriptors for the reference image
    refImg_kp, refImg_des = featureDetect.detectAndCompute(refImg, None)
    logger.info('Reference Image: found {} keypoints'.format(len(refImg_kp)))

    ### Loop over video frames ###############################################
    if nFrames and nFrames < totalFrames:
        framesToUse = np.arange(0, nFrames, 1)
    else:
        framesToUse = np.arange(0, totalFrames, 1)
    frameProcessing_startTime = time.time()
    frameCounter = 0
    
    while vid.isOpened():
        # read the next frame of the video
        ret, frame = vid.read()

        # check if it's a valid frame
        if (ret is True) and (frameCounter in framesToUse):

            # make copy of the reference image for later use
            ref_frame = refImgColor.copy()

            # process this frame
            processedFrame = processFrame(frame,
                                          frameCounter,
                                          refImg_kp,
                                          refImg_des,
                                          featureDetect)
            
            # if good match between reference image and this frame
            if processedFrame['foundGoodMatch']:

                # grab the gaze data (world coords) for this frame
                thisFrame_gazeData_world = gazeWorld_df.loc[gazeWorld_df['frame_idx'] == frameCounter]

                # project the reference image back into the video as a way to check for good mapping
                ref2world_frame = projectImage2D(processedFrame['origFrame'], processedFrame['ref2world'], refImgColor)

                # loop over all gaze data for this frame, translate to different coordinate systems
                for i, gazeRow in thisFrame_gazeData_world.iterrows():
                    timeinms = gazeRow['seconds']
                    ts = gazeRow['timestamp']
                    conf = gazeRow['confidence']

                    # translate normalized gaze data to world pixel coords
                    world_gazeX = gazeRow['norm_pos_x'] * processedFrame['frame_gray'].shape[1]
                    world_gazeY = gazeRow['norm_pos_y'] * processedFrame['frame_gray'].shape[0]

                    # covert from world to reference image pixel coordinates
                    ref_gazeX, ref_gazeY = mapCoords2D((world_gazeX, world_gazeY), processedFrame['world2ref'])

                    # create dict for this row
                    thisRow_df = pd.DataFrame({'timeinms':timeinms/1000,
                                               'gaze_ts': ts,
                                               'worldFrame': frameCounter,
                                               'confidence': conf,
                                               'world_gazeX': round(world_gazeX),
                                               'world_gazeY': round(world_gazeY),
                                               'ref_gazeX': round(ref_gazeX),
                                               'ref_gazeY': round(ref_gazeY)},
                                               index=[i])

                    # append row to gazeMapped_df output
                    if 'gazeMapped_df' in locals():
                        gazeMapped_df = pd.concat([gazeMapped_df, thisRow_df])
                    else:
                        gazeMapped_df = thisRow_df

                    ### Draw gaze circles on frames
                    if i == thisFrame_gazeData_world.index.max():
                        dotColor = [96, 52, 234]            # pinkish/red
                        dotSize = 12
                    else:
                        dotColor = [168, 231, 86]            # minty green
                        dotSize = 8

                    # world frame
                    cv2.circle(frame,
                               (int(world_gazeX), int(world_gazeY)),
                               dotSize,
                               dotColor,
                               -1)

                    # ref frame
                    cv2.circle(ref_frame,
                               (int(ref_gazeX), int(ref_gazeY)),
                               dotSize,
                               dotColor,
                               -1)
            else:
                # if not a good match, use the original frame for the ref2world
                ref2world_frame = processedFrame['origFrame']
                
                # grab the gaze data (world coords) for this frame
                thisFrame_gazeData_world = gazeWorld_df.loc[gazeWorld_df['frame_idx'] == frameCounter]
                
                # loop over all gaze data for this frame, translate to different coordinate systems
                for i, gazeRow in thisFrame_gazeData_world.iterrows():
                    timeinms = gazeRow['seconds']
                    ts = gazeRow['timestamp']
                    conf = gazeRow['confidence']
                    # translate normalized gaze data to world pixel coords
                    world_gazeX = gazeRow['norm_pos_x'] * frame.shape[1]
                    world_gazeY = gazeRow['norm_pos_y'] * frame.shape[0]
                    # create dict for this row
                    thisRow_df = pd.DataFrame({'timeinms':timeinms/1000,
                                               'gaze_ts': ts,
                                               'worldFrame': frameCounter,
                                               'confidence': conf,
                                               'world_gazeX': round(world_gazeX),
                                               'world_gazeY': round(world_gazeY),
                                               'ref_gazeX': "",
                                               'ref_gazeY': ""},
                                               index=[i])

                    # append row to gazeMapped_df output
                    if 'gazeMapped_df' in locals():
                        gazeMapped_df = pd.concat([gazeMapped_df, thisRow_df])
                    else:
                        gazeMapped_df = thisRow_df

            # write outputs to video
            vidOut_world.write(frame)
            vidOut_ref.write(ref_frame)
            vidOut_ref2world.write(ref2world_frame)
            
        else:
            print("here coming")
        # increment frame counter
        frameCounter += 1
        if frameCounter > np.max(framesToUse):
            # release all videos
            vid.release()
            vidOut_world.release()
            vidOut_ref.release()
            vidOut_ref2world.release()
            print(gazeMapped_df)
            # write out gaze data
            try:
                colOrder = ['worldFrame', 'timeinms', 'gaze_ts', 'confidence',
                            'world_gazeX', 'world_gazeY',
                            'ref_gazeX', 'ref_gazeY']
                gazeMapped_df[colOrder].to_csv(join(outputDir, 'gazeData_mapped.tsv'),
                                               sep='\t',
                                               index=False,
                                               float_format='%.3f')
                
            except Exception as e:
                logger.info(e)
                logger.info('cound not write gazeData_mapped to csv')
                pass

    endTime = time.time()
    frameProcessing_time = endTime - frameProcessing_startTime
    logger.info('Total time: %s seconds' % frameProcessing_time)
    logger.info('Avg time/frame: %s seconds' % (frameProcessing_time / framesToUse.shape[0]))


def processFrame(frame, frameIdx, ref_kp, ref_des, featureDetect):
    """ Process single frame from the world camera to determine mapping to
    ref image

    Parameters
    ---------
    frame : np.ndarray
        frame from world camera video
    frameIdx : int
        frame index (0-based)
    ref_kp : list
        identified keypoints on the reference image
    ref_des : np.ndarray
        descriptors for the reference image keypoints
    featureDetect : object
        instance of cv2 SIFT class

    Returns
    -------
    fr : dict
        dictionary with entries storing all of the relevant output for this
        particular frame

    """
    logger = logging.getLogger()

    fr = {}        # create dict to store info for this frame

    # create copy of original frame
    origFrame = frame.copy()
    fr['origFrame'] = origFrame         # store

    # convert to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fr['frame_gray'] = frame_gray

    # try to match the frame and the reference image
    try:
        frame_kp, frame_des = featureDetect.detectAndCompute(frame_gray, None)
        logger.info('found {} features on frame {}'.format(len(frame_kp), frameIdx))

        if len(frame_kp) < 2:
            ref_matchPts = None
        else:
            ref_matchPts, frame_matchPts = findMatches(ref_kp,
                                                       ref_des,
                                                       frame_kp,
                                                       frame_des)

        # check if matches were found
        try:
            numMatches = ref_matchPts.shape[0]

            # if sufficient number of matches....
            if numMatches > 0:
                logger.info('found {} matches on frame {}'.format(numMatches, frameIdx))
                sufficientMatches = True
            else:
                logger.info('Insufficient matches ({}} matches) on frame {}'.format(numMatches, frameIdx))
                sufficientMatches = False

        except:
            print('no matches found on frame {}'.format(frameIdx))
            sufficientMatches = False
            pass

        fr['foundGoodMatch'] = sufficientMatches

        # figure out homographies between coordinate systems
        if sufficientMatches:
            ref2world_transform, mask = cv2.findHomography(ref_matchPts.reshape(-1, 1, 2),
                                                           frame_matchPts.reshape(-1, 1, 2),
                                                           cv2.RANSAC,
                                                           5.0)
            world2ref_transform = cv2.invert(ref2world_transform)

            fr['ref2world'] = ref2world_transform
            fr['world2ref'] = world2ref_transform[1]

    except:
        fr['foundGoodMatch'] = False

    # return the processed frame
    return fr


def addMissingTime(outputFile, rawFile):
    gaze_mapped_df = pd.read_table(outputFile, sep='\t')
    gaze_raw_df = pd.read_table(rawFile, sep='\t')
    gaze_mapped_df['timeinms'] = gaze_mapped_df['timeinms']/1000
    gaze_raw_df['seconds'] = gaze_raw_df['seconds']/1000
    missing_rows = gaze_raw_df.loc[~gaze_raw_df['seconds'].isin(gaze_mapped_df['timeinms']), ['seconds', 'index', 'gaze_pos_x', 'gaze_pos_y']]
    print(len(missing_rows))
    # Rename mapping (old_name -> new_name)
    rename_mapping = {
        'seconds': 'timeinms',
        'index': 'gaze_ts',
        'gaze_pos_x': 'world_gazeX',
        'gaze_pos_y': 'world_gazeY'    
    }
    missing_rows = missing_rows.rename(columns=rename_mapping)
    
    # Ensure all columns from gaze_mapped_df exist in missing_rows
    for col in gaze_mapped_df.columns:
        if col not in missing_rows.columns:
            missing_rows[col] = None  # Fill missing columns with empty values

    # Merge both DataFrames
    merged_df = pd.concat([gaze_mapped_df, missing_rows], ignore_index=True)
    merged_df = merged_df.sort_values(by='timeinms', ascending=True).reset_index(drop=True)
    try:
        colOrder = ['worldFrame', 'timeinms', 'gaze_ts', 'confidence',
                    'world_gazeX', 'world_gazeY',
                    'ref_gazeX', 'ref_gazeY']
        merged_df[colOrder].to_csv(join(outputDir, 'gazeData_mapped.tsv'),
                                        sep='\t',
                                        index=False,
                                        float_format='%.3f')
        
    except Exception as e:
        logger.info(e)
        logger.info('cound not write gazeData_mapped to csv')
        pass
    

if __name__ == '__main__':
    # to run file use following command: python mapGaze.py myGazeFile.csv Glass2_1.mp4 myReferenceImage.jpeg

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder',
                         help='path to data file')
    parser.add_argument('referenceImage',
                        help='path to reference image file')
    
    args = parser.parse_args()
    input_folder = args.input_folder
    gazeData = f"{input_folder}/preprocessed_files/gazeData_world.tsv"
    worldCameraVid = f"{input_folder}/preprocessed_files/worldCamera.mp4"
    
    
    # Input error checking
    badInputs = []
    for arg in [gazeData, worldCameraVid, args.referenceImage]:
        if not os.path.exists(arg):
            badInputs.append(arg)
    if len(badInputs) > 0:
        [print('{} does not exist! Check your input file path'.format(x)) for x in badInputs]
        sys.exit()
        
    # Set output directory
    if input_folder is not None:
        outputDir = join(input_folder, 'mappedGazeOutput_feature_detection')
    else:
        outputDir = 'mappedGazeOutput_feature_detection'
    
    ## process the recording
    print('processing the recording...')
    print('Output saved in: {}'.format(outputDir))
    processRecording(gazeData=gazeData,
                     worldCameraVid=worldCameraVid,
                     referenceImage=args.referenceImage,
                     outputDir=outputDir)
    addMissingTime(outputFile=f"{outputDir}/gazeData_mapped.tsv",
                   rawFile=f"{input_folder}/preprocessed_files/gazeData_raw.tsv")
    