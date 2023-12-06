# Computer Vision for Engineers Group Project
## Overview
This project integrates various computer vision and image processing techniques to process and analyze data from multiple sources, including depth sensors and phone cameras. 
The main functionalities include pose estimation, pixel-to-world coordinate transformation, and video processing. 
It uses Python libraries like OpenCV and NumPy, along with custom modules like coord_transform, pose, and pixel_to_3d.

## Installation
### Prerequisites
Python 3.x /n
OpenCV (cv2) /n
NumPy /n
[Any other dependencies that your project requires]
### Steps
- Clone the repository or download the source code.
- Install the required Python packages
- Place the required data files (.bag and .mp4 files) in the specified directory.
## Usage
- Setting Up Data Files: Before running the script, ensure that the data files (.bag and .mp4 files) are placed in the correct directory. Modify the color_file, depth_file, and phone_file variables in the script to point to the correct file paths.

- Running the Script: Execute the script in a Python environment. The script will process the provided data files and generate output based on the pose estimation and image processing algorithms.

- Output: The script outputs a video file combining the processed data from both the depth sensor and the phone camera.

## Custom Modules
- coord_transform: Module for transforming coordinates.
- pose: Pose Estimation module based on MoveNet.
- pixel_to_3d: Module for converting pixel coordinates to 3D world coordinates.
- extract_data: Utility for extracting data from sensor files.
- camera_merge: Module for merging video from different camera sources.
  
## Note
The script includes parameters for camera calibration and transformation matrices that might need adjustment based on your specific hardware setup.
The script is configured for a specific resolution and camera parameters, so it may require modifications to work with different setups.
Contributing
Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
