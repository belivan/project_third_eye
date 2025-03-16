import numpy as np
import cv2 as cv
from coord_transform import coord2point
from pose import PoseEstimator
from pixel_to_3d import PixelToWorldProcessor
import os
from extract_data import extract_data
from camera_merge import VideoMaker


# |------------------------------------------------------------------------------|
# |                              Initialise parameters                           |
# |------------------------------------------------------------------------------|

# Gets the directory where the script is located
current_dir = os.path.dirname(__file__)
os.chdir(current_dir)

# SPECIFY DATA FILES HERE: **CHANGE DIRECTORY IF NECESSARY**
color_file, depth_file, phone_file = extract_data("second_sample.bag", "origin.mp4")

# INITIALIZE DATA PROCESSORS

pose = PoseEstimator("movenet_thunder_f16")
pixel_to_world = PixelToWorldProcessor()
video = VideoMaker(output_filename="result.mp4")

# -------------------------DEFINE PHONE PARAMETERS----------------------

# Rotation matrix
R = np.array([[0.0, 0.0, -1.0],
              [0.0, 1.0, 0.0],
              [1.0, 0.0, 0.0]])
# Translation matrix
t = np.array([[4.2672 * np.cos(np.pi / 4),
               0.0,
               4.2672 * np.cos(np.pi / 4) + 0.860]]).T

# intrinsics parameters of the phone
f = 0.00277  # meters
delta = 1.4e-6  # meters
Cu = 4032 / 2
Cv = 3024 / 2

A = np.array([[f / delta, 0.0, Cu],
              [0.0, f / delta, Cv],
              [0.0, 0.0, 1.0]])

# ---------------------------Load phone files-----------------------------

phone_images = np.load(phone_file)
depth_images = np.load(depth_file)
color_images = np.load(color_file)


video.setup_video_writer(phone_images[0], color_images[0])

# |----------------------------------------------------------------------|
# |                               Main loop                              |
# |----------------------------------------------------------------------|

for i in range(len(color_images)):
    # Get all images as NumPy arrays
    phone_image = phone_images[i]
    depth_image = depth_images[i]
    sense_image = color_images[i]

    # Get pose keypoints from color sensor image
    pose_keypoints = pose.predict_keypoints_transform(image=sense_image)
    # print("Successfully created pose keypoints for frame " + str(i) + ".")

    if None not in pose_keypoints:
        # Get 3D coordinates from depth image
        # Assuming pose_keypoints is a list of [x, y] pairs
        depth_coords = []
        for pose_keypoint in pose_keypoints:
            cv.circle(sense_image, pose_keypoint, 5, (255, 0, 0), -1)
            # Check if the conditions are met
            if pose_keypoint[0] < 1280 and pose_keypoint[1] < 720:
                # Get 3D coordinates from depth image
                depth_coord = pixel_to_world.get_3D_point(depth_image, pose_keypoint[0], pose_keypoint[1])
                depth_coords.append(depth_coord)
            else:
                # If conditions are not met, just forget it
                continue

        # Get phone coordinates from 3D coordinates
        phone_coords = coord2point(depth_coords, R, t, A)

        for coord in phone_coords:
            coord[0] = 1280 / 4032 * coord[0]
            coord[1] = 720 / 3024 * coord[1]
            cv.circle(phone_image, coord, 3, (0, 0, 255), -1)

    # cv.imshow("phone image", phone_image)
    # cv.imshow("color image", cv.cvtColor(sense_image, cv.COLOR_RGB2BGR))

    # Write frame to video
    video.write_frame(phone_image, sense_image, depth_image, stream=False)
    path = "/Users/deniskaanalpay/Desktop/CMU/Term 1/24678 Computer Vision/CV_Project/output/"
    cv.imwrite(path + "/phone_image_" + str(i) + ".png", phone_image)
    cv.imwrite(path + "/color_image_" + str(i) + ".png", cv.cvtColor(sense_image, cv.COLOR_RGB2BGR))
    depth_clipped = np.clip(depth_image, 0, 8000)
        # Normalize the depth image to fall within the range 0-255
    depth_normalized = cv.normalize(depth_clipped, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)

        # Apply the color map to the normalized depth image
    depth_image = cv.applyColorMap(depth_normalized, cv.COLORMAP_JET)
    depth_image_jet = cv.applyColorMap(cv.normalize(depth_image, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U), cv.COLORMAP_JET)
    cv.imwrite(path + "/depth_image_" + str(i) + ".png", depth_image_jet)

    
    if cv.waitKey(100) == ord("q"):
        break

# video.close()
print("Video processing complete.")
cv.destroyAllWindows()
