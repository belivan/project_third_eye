import numpy as np
import pyrealsense2 as rs


class PixelToWorldProcessor:
    def __init__(self):
        self.depth_scale = 0.001  # Default value for Intel Realsense D435i
        self.depth_intrin = self.set_manual_depth_intrinsics()

    def get_3D_point(self, depth_image, pixel_x, pixel_y):
        """
            Get 3D point from depth image and pixel coordinates

            Args:
                depth_image (array): Depth image
                pixel_x (int): Pixel x-coordinate
                pixel_y (int): Pixel y-coordinate

            Returns:
                array: 3D point

            Raises:
                None

            Examples:
                depth_image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
                pixel_x = 1
                pixel_y = 1
                get_3D_point(depth_image, pixel_x, pixel_y)
                # Output: [0.0, 0.0, 5.0]
            """
        depth = depth_image[pixel_y, pixel_x] * self.depth_scale
        if depth == 0:
            return None
        return rs.rs2_deproject_pixel_to_point(
            self.depth_intrin, [pixel_x, pixel_y], depth)

    def get_3D_point_from_index(self, index, pixel_x, pixel_y):
        """
            Get 3D point from depth image and pixel coordinates

            Args:
                index (int): Index of the depth image
                pixel_x (int): Pixel x-coordinate
                pixel_y (int): Pixel y-coordinate

            Returns:
                array: 3D point

            Raises:
                None

            Examples:
                # Example usage
                point = get_3D_point_from_index(0, 100, 200)
            """
        depth_image = self.data.get_depth_image(index)

        if depth_image is None:
            return "Error: depth image is None"

        depth = depth_image[pixel_y, pixel_x] * self.depth_scale
        if depth == 0:
            return None
        return rs.rs2_deproject_pixel_to_point(
            self.depth_intrin, [pixel_x, pixel_y], depth)

    def get_3D_points(self, depth_image, pixel_coordinates):
        """
            Get 3D points from depth image and pixel coordinates

            Args:
                depth_image (array): Depth image
                    The depth image from which to extract the 3D points.
                pixel_coordinates (array): Array of pixel coordinates
                    The array of pixel coordinates representing the points in the depth image.

            Returns:
                array: 3D points
                    The array of 3D points corresponding to the given pixel coordinates.
            """
        depth = depth_image[pixel_coordinates[:, 1],
        pixel_coordinates[:, 0]] * self.depth_scale

        if depth is None:
            return "Error: in get_3D_points depth is None"

        depth = np.expand_dims(depth, axis=1)
        pixel_coordinates = np.concatenate((pixel_coordinates, depth), axis=1)
        return rs.rs2_deproject_pixel_to_point(
            self.depth_intrin, pixel_coordinates)

    def get_3D_points_from_index(self, index, pixel_coordinates):
        """
            Get 3D points from depth image and pixel coordinates

            Args:
                index (int): Index of the depth image
                pixel_coordinates (array): Array of pixel coordinates

            Returns:
                array: 3D points
            """
        depth_image = self.data.get_depth_image(index)

        if depth_image is None:
            return "Error: depth image is None"

        if pixel_coordinates is None:
            return "Error: pixel coordinates is None"

        depth = depth_image[pixel_coordinates[:, 1],
        pixel_coordinates[:, 0]] * self.depth_scale
        depth = np.expand_dims(depth, axis=1)
        pixel_coordinates = np.concatenate(
            (pixel_coordinates, depth), axis=1)
        return rs.rs2_deproject_pixel_to_point(
            self.depth_intrin, pixel_coordinates)
    
    def set_manual_depth_intrinsics(self):
        """
        Set manual intrinsics for Intel Realsense D435i

        Returns:
            rs.intrinsics: Intel Realsense intrinsics
        """
        # Instrinsics for Intel Realsense D435i (aligned frames)
        intrin = rs.intrinsics()
        intrin.width = 1280
        intrin.height = 720
        intrin.ppx = 636.2242431640625
        intrin.ppy = 351.17333984375
        intrin.fx = 912.1310424804688
        intrin.fy = 912.2687377929688
        intrin.model = rs.distortion.inverse_brown_conrady
        intrin.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
        return intrin
