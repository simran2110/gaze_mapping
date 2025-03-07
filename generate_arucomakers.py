import cv2
import numpy as np
import cv2.aruco as aruco

def generate_aruco_markers(count=100, marker_size=200, dictionary=aruco.DICT_6X6_1000, save_dir="aruco_markers"):
    """Generates multiple unique ArUco markers and saves them as images.
    
    Args:
        count (int): Number of unique ArUco markers to generate.
        marker_size (int): Size of each marker in pixels (default: 200x200).
        dictionary (int): ArUco dictionary type (default: DICT_6X6_250).
        save_dir (str): Directory to save the marker images.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    aruco_dict = aruco.getPredefinedDictionary(dictionary)
    
    for marker_id in range(count):
        marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
        marker_img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        save_path = os.path.join(save_dir, f"aruco_marker_{marker_id}.png")
        cv2.imwrite(save_path, marker_img)
        
    print(f"Generated {count} ArUco markers in {save_dir}")

# Generate 100 unique ArUco markers
generate_aruco_markers(250)