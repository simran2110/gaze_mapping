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

def detect_markers_id_and_corner(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    
    parameters = cv2.aruco.DetectorParameters()
    
    # Create the ArUco detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # # Detect the markers in the image
    corners, ids, rejected = detector.detectMarkers(gray)


    detected_marker_corners = {}
    if len(corners) > 0:

        for i in range(len(ids)):
            # Get the four corner points of the marker
            marker_corners = corners[i][0]

            # Extract coordinates of each corner
            top_left = tuple(marker_corners[0])
            top_right = tuple(marker_corners[1])
            bottom_right = tuple(marker_corners[2])
            bottom_left = tuple(marker_corners[3])
            # Print coordinates
            #print(f"Marker ID {ids[i][0]}: TL{top_left}, TR{top_right}, BR{bottom_right}, BL{bottom_left}")
            detected_marker_corners[int(ids[i][0])] = [top_left, top_right, bottom_right, bottom_left]
            # video_points.append((corner[0][0], corner[0][1]))
    return detected_marker_corners


def find_homography(frame_points, ref_points):
    # Reshape to (N, 1, 2) as required by OpenCV
    frame_points = frame_points.reshape(-1, 1, 2)
    ref_points = ref_points.reshape(-1, 1, 2)

    # Check if enough points are detected
    if len(frame_points) >= 4 and len(ref_points) >= 4:
        fr = {}
        
        # Compute Homography Matrix
        ref2world_transform, mask = cv2.findHomography(ref_points, frame_points, cv2.RANSAC, 5.0)
        world2ref_transform = cv2.invert(ref2world_transform)
        fr['ref2world'] = ref2world_transform
        fr['world2ref'] = world2ref_transform[1]
        return fr
    else:
        return None
        

# Function to map gaze data (world coordinates) to the reference image coordinates
def map_gaze_to_reference(gaze_data, homography):
    if homography is None:
        return None
    
    # Convert world coordinates to image coordinates using homography
    world_coords = np.array([gaze_data], dtype="float32")
    world_coords = world_coords.reshape((-1, 1, 2))

    reference_coords = cv2.perspectiveTransform(world_coords, homography)

    return reference_coords[0][0]  # Returns the mapped (x, y) on reference image

# Main function to process video, map gaze data, and save output
def process_video(gaze_data_file, video_file, reference_image_file, outputDir):
    # Load gaze data from file (assuming TSV format)
    gaze_data = pd.read_csv(gaze_data_file, sep='\t')

    # Load video file
    cap = cv2.VideoCapture(video_file)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load reference image
    reference_image = cv2.imread(reference_image_file)
    reference_image_pts = detect_markers_id_and_corner(reference_image)
    
    # Create output directory if not exists
    os.makedirs(outputDir, exist_ok=True)

    frameCounter = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('output_video.mp4', fourcc, frame_rate, (frame_width, frame_height))

    # Iterate through each frame of the video
    frameCounter = 0
    gaze_idx = 0
    homography = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        # Convert to grayscale
        frame_pts = detect_markers_id_and_corner(frame)
        
        # Get only the common marker IDs
        common_ids = sorted(set(frame_pts.keys()) & set(reference_image_pts.keys()))

        # Extract points in the same order for both images
        frame_points = np.array([frame_pts[k] for k in common_ids], dtype="float32")
        ref_points = np.array([reference_image_pts[k] for k in common_ids], dtype="float32")
        
        # grab the gaze data (world coords) for this frame
        thisFrame_gazeData_world = gaze_data.loc[gaze_data['frame_idx'] == frameCounter]
                
        transformation_positive = False       
        print(frameCounter)
        # Check if enough points are detected
        if len(frame_points) >= 4 and len(ref_points) >= 4:
            homography =  find_homography(frame_points, ref_points)
            # Get gaze data for current frame
            if homography is not None and gaze_idx < len(gaze_data):
                print("Found markers")
                # loop over all gaze data for this frame, translate to different coordinate systems
                for i, gazeRow in thisFrame_gazeData_world.iterrows():
                    timeinms = gazeRow['seconds']
                    ts = gazeRow['timestamp']
                    conf = gazeRow['confidence']

                    # translate normalized gaze data to world pixel coords
                    world_gazeX = gazeRow['norm_pos_x'] * frame.shape[1]
                    world_gazeY = gazeRow['norm_pos_y'] * frame.shape[0]

                    # covert from world to reference image pixel coordinates
                    ref_gazeX, ref_gazeY = map_gaze_to_reference((world_gazeX, world_gazeY), homography['world2ref'])

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
                    transformation_positive = True
            
        if not transformation_positive:
            print("NOT Found markers")
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
                                            'world_gazeX': world_gazeX,
                                            'world_gazeY': world_gazeY,
                                            'ref_gazeX': "",
                                            'ref_gazeY': ""},
                                            index=[i])

                # append row to gazeMapped_df output
                if 'gazeMapped_df' in locals():
                    gazeMapped_df = pd.concat([gazeMapped_df, thisRow_df])
                else:
                    gazeMapped_df = thisRow_df 
            
            
        # Write the processed frame to the output video
        output_video.write(frame)

        frameCounter += 1

    # Release video capture and writer
    cap.release()
    output_video.release()

    print(f"Video processing complete. Output saved to {outputDir}/output_video.mp4v")
    
    try:
        colOrder = ['worldFrame', 'timeinms', 'gaze_ts', 'confidence',
                    'world_gazeX', 'world_gazeY',
                    'ref_gazeX', 'ref_gazeY']
        gazeMapped_df[colOrder].to_csv(join(outputDir, 'gazeData_mapped.tsv'),
                                        sep='\t',
                                        index=False,
                                        float_format='%.3f')
        
    except Exception as e:
        print('cound not write gazeData_mapped to csv')
        pass
    
def addMissingTime(outputFile, rawFile):
    gaze_mapped_df = pd.read_table(outputFile, sep='\t')
    gaze_raw_df = pd.read_table(rawFile, sep='\t')
    gaze_raw_df['seconds'] = gaze_raw_df['seconds']/1000
    missing_rows = gaze_raw_df.loc[~gaze_raw_df['seconds'].isin(gaze_mapped_df['timeinms']), ['seconds', 'index', 'gaze_pos_x', 'gaze_pos_y']]
    
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
        outputDir = join(input_folder, 'mappedGazeOutput_markers_detection_detection')
    else:
        outputDir = 'mappedGazeOutput_markers_detection_detection'
    
    ## process the recording
    print('processing the recording...')
    print('Output saved in: {}'.format(outputDir))
    print(f"gazeData={gazeData}, worldCameraVid={worldCameraVid}, referenceImage={args.referenceImage},outputDir={outputDir}")
    process_video(gazeData, worldCameraVid, args.referenceImage, outputDir)
    addMissingTime(outputFile=f"{outputDir}/gazeData_mapped.tsv",
                   rawFile=f"{input_folder}/preprocessed_files/gazeData_raw.tsv")
    