# Gaze-mapping

A Python file for mapping mobile gaze data to a fixed target stimulus.

# Installation
The `gaze_map` tool has been built and tested using `Python 3.12`. To install required dependencies, navigate to the root of this repository and use:

> pip install -r requirements.txt

# Overview

Eye-trackers allow for measures like gaze position to be recorded under naturalistic conditions where an individual is free to move around. Gaze position is typically recorded relative to an outward facing camera attached to the eye-tracker and approximating the point-of-view of the individual wearing the device. As such, gaze position is recorded relative to the individual's position and orientation, which changes as the participant moves. Since gaze position is recorded without any reference to fixed objects in the environment, this poses a challenge for studying how an individual views a particular stimulus over time.

This toolkit addresses this challenge by automatically identifying the target stimulus on every frame of the recording and mapping the gaze positions to a fixed representation of the stimulus. At the end, gaze positions across the entire recording are expressed in pixel coordinates of the fixed 2D target stimulus.

# Usage Guide
Steps followed to map the gaze data:
1) Preprocessing of raw data from tobii pro:
    The raw data from the tobii pro recording are send to the preprocessing python file to get a tsv with each row of data mapped to frame number. This preprocessed data is required by map_gaze.py script to efficiently perform mapping.

    To feed the files as input, create a folder with the required files (with any name) with following files:
        Eye tracking video
        segment.json file
        livedata.json.gz 
        These files are obtained as an output from tobii controller.

    Python command to run the file:
        python <preprocessing_python_script> <folder_name>
        
    Example of command using the example data given:
        python preprocessing.py markers_data/

2) Mapping the gaze data to static snapshot:
    The script is designed to use the preprocessed tsv file(i.e. generated as output from last step) that contains frame number corresponding to each timestamp. Each video frame is iterated to detect Aruco markers(placed on edges of each screens). If a sufficient number of matches between the frame and the reference image were found, a homography transformation was applied to align the frame with the reference image. Once the transformation was established, gaze points from that frame were projected onto their corresponding locations in the static snapshot, ensuring accurate placement.
    If no Aruco markers are detected in a frame, a feature-based matching technique(SIFT) is applied. Key feature points were extracted from the frame and compared with the reference image using feature detection algorithms. If a sufficient number of feature matches were identified, a transformation was computed to align the frame with the static image. The gaze data from the frame was then mapped onto the corresponding locations in the reference image.
    If neither Aruco markers nor sufficient feature matches were found, the frame was discarded due to the inability to reliably align it with the static reference. 


    Python command to run the file:
        python <gaze_mapping_script> <folder_name> <path_of_reference_snapshot>
        
    Example of command using the example data given:
        python map_gaze.py markers_data reference_image.JPG

3) Final output:
    The final output generated accurately mapped (x, y) gaze coordinates for each timestamp, aligning gaze points with reference snapshots to enable analysis of visual attention patterns over time. 

