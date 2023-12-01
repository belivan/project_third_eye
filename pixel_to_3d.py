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
