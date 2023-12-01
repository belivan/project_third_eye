import numpy as np
import cv2 as cv
import os
from coord_transform import coord2point
from data_manager import DataProcessor
from pose import PoseEstimator
from pixel_to_3d import PixelToWorldProcessor
from camera_merge import SensePhoneMerger

class ImageProcessor:
    def __init__(self):
        """
        Constructor for the ImageProcessor class.
        Sets up the working directory and initializes the PoseEstimator and PixelToWorldProcessor.
        Also loads the images from specified paths.
        """
        current_dir = os.path.dirname(__file__)
        os.chdir(current_dir)

        self.pose = PoseEstimator("movenet_thunder_f16")
        self.pixel_to_world = PixelToWorldProcessor()

        # Initialize phone parameters
        self.init_phone_params()

        # Load images
        self.phone_images = np.load("data/origin_sync/phone_frames.npy")
        self.depth_images = np.load("data/depth_frames.npy")
        self.color_images = np.load("data/color_frames.npy")

    def init_phone_params(self):
        """
        Initializes the phone parameters such as rotation matrix, translation vector,
        and camera intrinsic parameters.
        """
        self.R = np.array([[0.0, 0.0, -1.0],
                           [0.0, 1.0, 0.0],
                           [1.0, 0.0, 0.0]])

        self.t = np.array([[4.2672 * np.cos(np.pi / 4),
                            0.0,
                            4.2672 * np.cos(np.pi / 4) + 0.860]]).T

        f = 0.00277  # focal length in meters
        delta = 1.4e-6  # pixel size in meters
        Cu = 4032 / 2  # principal point x-coordinate
        Cv = 3024 / 2  # principal point y-coordinate

        self.A = np.array([[f / delta, 0.0, Cu],
                           [0.0, f / delta, Cv],
                           [0.0, 0.0, 1.0]])

    def process_images(self):
        """
        Processes each image in the loaded image arrays.
        """
        for i in range(len(self.phone_images)):
            self.process_single_image(self.phone_images[i], self.depth_images[i], self.color_images[i])

    def process_single_image(self, phone_image, depth_image, sense_image):
        """
        Processes a single image, including pose estimation, depth coordinate computation,
        and phone coordinate transformation.
        """
        pose_keypoints = self.pose.predict_keypoints_transform(image=sense_image)
        print("Successfully created pose keypoints")

        if None not in pose_keypoints:
            depth_coords = self.get_depth_coords(depth_image, pose_keypoints)
            phone_coords = coord2point(depth_coords, self.R, self.t, self.A)
            self.draw_points(phone_image, phone_coords)

        self.display_images(phone_image, sense_image)

    def get_depth_coords(self, depth_image, pose_keypoints):
        """
        Calculates depth coordinates for each pose keypoint.
        """
        depth_coords = []
        for pose_keypoint in pose_keypoints:
            cv.circle(sense_image, pose_keypoint, 5, (255, 0, 0), -1)
            if pose_keypoint[0] < 1280 and pose_keypoint[1] < 720:
                depth_coord = self.pixel_to_world.get_3D_point(depth_image, pose_keypoint[0], pose_keypoint[1])
                depth_coords.append(depth_coord)
        return depth_coords

    def draw_points(self, phone_image, phone_coords):
        """
        Draws points on the phone image based on the transformed coordinates.
        """
        for coord in phone_coords:
            coord[0] = 1280 / 4032 * coord[0]
            coord[1] = 720 / 3024 * coord[1]
            cv.circle(phone_image, coord, 5, (0, 0, 255), -1)

    def display_images(self, phone_image, sense_image):
        """
        Displays the processed phone image and color image.
        """
        cv.imshow("phone image", phone_image)
        cv.imshow("color image", cv.cvtColor(sense_image, cv.COLOR_RGB2BGR))
        if cv.waitKey(100) == ord("q"):
            cv.destroyAllWindows()

# Usage
image_processor = ImageProcessor()
image_processor.process_images()
