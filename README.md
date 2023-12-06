# Computer Vision for Engineers Group Project
## Overview
This project integrates various computer vision and image processing techniques to process and analyze data from multiple sources, including depth sensors and phone cameras. 
The main functionalities include pose estimation, pixel-to-world coordinate transformation, and video processing. 
It uses Python libraries like OpenCV and NumPy, along with custom modules like coord_transform, pose, and pixel_to_3d.

## Installation
### Prerequisites
Python 3.x
OpenCV (cv2)
NumPy
PyRealSense2
TensorFlow
TensorFlowHub
### Steps
- Clone the repository or download the source code.
- Install the required Python packages
- Place the required data files (.bag and .mp4 files) in the specified directory.
## Usage
- Setting Up Data Files: Before running the script, ensure that the data files (.bag and .mp4 files) are placed in the correct directory. In the main.py modify the color_file, depth_file, and phone_file variables in the script to point to the correct file paths.
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

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

### About the GPL License

The GNU General Public License is a free, copyleft license for software and other kinds of works. The licenses for most software and other practical works are designed to take away your freedom to share and change the works. By contrast, the GNU General Public License is intended to guarantee your freedom to share and change all versions of a program--to make sure it remains free software for all its users.

Under this license, anyone who modifies and redistributes the project must also distribute their modifications under the same license. This requirement ensures that modifications and extensions to the project remain open-source and available to the community under the same terms.

For more information on the GNU General Public License v3.0, please visit [https://www.gnu.org/licenses/gpl-3.0.en.html](https://www.gnu.org/licenses/gpl-3.0.en.html).
