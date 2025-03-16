import numpy as np
import os
import cv2


class VideoMaker:
    def __init__(self, output_directory=None, output_filename=None):
        """
        Initialize the CameraMerge object.
        You can create a video output file by specifying the output
        directory and output filename.

        Video Format Options: .avi, .mp4, .mkv, .mov
        Example Usage:  CameraMerge(output_directory="output",
                                    output_filename="merged.avi")

        Args:
            output_directory (str, optional): The directory where the video
            output file will be saved.
            Defaults to None.
            output_filename (str, optional): The name of the video output file.
            Defaults to None.
        """
        self.output_filename = output_filename
        self.output_directory = output_directory if output_directory is not None else "."

        if self.output_filename:
            self.video_path = os.path.join(
                self.output_directory, self.output_filename)

        self.fps_phone = float(30.0)
        self.fps_sense = float(30.0)

        self.video_writer = None

        self.frame_index = 0

    def setup_video_writer(self, example_img_phone, example_img_sense):
        """
        Sets up a video writer.

        Returns:
            cv2.VideoWriter: The video writer object.
        """
        if example_img_sense.shape != example_img_phone.shape:
            example_img_sense = cv2.resize(example_img_sense,
                                           (example_img_phone.shape[1],
                                            example_img_phone.shape[0]))
        resized = cv2.resize(example_img_phone,
                             (example_img_phone.shape[1] * 2,
                              example_img_phone.shape[0] * 2))

        combined_temp = np.hstack((example_img_sense,
                                   example_img_sense))

        combined_example = np.vstack((resized, combined_temp))
        height, width, _ = combined_example.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 30, (width, height))

        if self.output_filename:
            os.makedirs(self.output_directory, exist_ok=True)
            print(f"Video writer to file: {self.video_path}")

    def write_frame(self, image_phone, image_color, image_depth, stream=True):
        """
        Perform comparison between iPhone and RealSense camera streams.

        Args:
            stream (bool, optional): Flag to enable/disable streaming.
            Defaults to True.
        """
        output_video = self.video_writer
        self.frame_index += 1

        if stream:
            cv2.namedWindow('iPhone and RealSense Stream', cv2.WINDOW_NORMAL)

        # image_phone = cv2.cvtColor(image_phone, cv2.COLOR_RGB2BGR)
        image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)

        # Clip the depth image to fall within the range 0-8000
        depth_clipped = np.clip(image_depth, 0, 8000)
        # Normalize the depth image to fall within the range 0-255
        depth_normalized = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        # Apply the color map to the normalized depth image
        image_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        if stream or self.output_filename:
            height, width = image_phone.shape[:2]
            resized_phone_image = cv2.resize(image_phone,
                                             (width * 2, height * 2))

            combined_temp = np.hstack((image_color, image_depth))

            if combined_temp.shape[1] != resized_phone_image.shape[1]:
                # Resize combined_temp to match the width of resized_phone_image
                combined_temp = cv2.resize(combined_temp,
                                           (resized_phone_image.shape[1],
                                            combined_temp.shape[0]))

            combined = np.vstack((resized_phone_image, combined_temp))

            # Position for the text (top-left corner)
            text_position = (10, 30)  # (x, y) coordinates

            # Font settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (0, 0, 0)
            font_thickness = 4

            # Add text to the frame
            cv2.putText(combined,
                        f'Frame: {self.frame_index}',
                        text_position, font, font_scale,
                        font_color, font_thickness)

            # Write to video if output path is set
            if self.output_filename:
                output_video.write(combined)
        path = "/Users/deniskaanalpay/Desktop/CMU/Term 1/24678 Computer Vision/CV_Project/output/"
        cv2.imwrite(path + "combined_image_" + str(self.frame_index) + ".png", combined)
        if stream:
            cv2.imshow('iPhone and RealSense Stream', combined)
            

        if self.frame_index % 50 == 0:
            print(f"Processed {self.frame_index} frames")

    def close(self):
        print("Closing video writer...")
        print(f"Total frames processed: {self.frame_index} frames && Total video duration: {self.frame_index / 30} seconds")
        self.video_writer.release()