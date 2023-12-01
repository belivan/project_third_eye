import numpy as np
import os
import cv2
from data_manager import DataProcessor


class SensePhoneMerger:
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
        self.data = DataProcessor()
        self.output_filename = output_filename
        self.output_directory = output_directory if output_directory is not None else "."

        if self.output_filename:
            self.video_path = os.path.join(
                self.output_directory, self.output_filename)

        self.fps_phone = float(30.0)
        self.total_duration = self.data.frames / self.fps_phone

        self.fps_sense = float(self.data.frames /
                               self.total_duration)

        self.interval_phone = int(1000 / self.fps_phone)
        self.interval_sense = int(1000 / self.fps_sense)

    def setup_video_writer(self):
        """
        Sets up a video writer.

        Returns:
            cv2.VideoWriter: The video writer object.
        """
        example_img_phone = self.data.get_phone_image(0)
        example_img_sense = self.data.get_color_image(0)

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
        return cv2.VideoWriter(self.video_path, fourcc,
                               self.fps_phone, (width, height))

    def perform_comparison(self, stream=True):
        """
        Perform comparison between iPhone and RealSense camera streams.

        Args:
            stream (bool, optional): Flag to enable/disable streaming.
            Defaults to True.
        """
        if self.output_filename and self.output_directory is not None:
            os.makedirs(self.output_directory, exist_ok=True)
            output_video = self.setup_video_writer()
            print(f"Video writer to file: {self.video_path}")

        if stream:
            cv2.namedWindow('iPhone and RealSense Stream', cv2.WINDOW_NORMAL)
        
        frames = self.data.frames

        for frame_index in range(frames):
            # Later, this will be replaced with the augmented phone image
            image_phone = self.data.get_phone_image(frame_index)
            image_phone = cv2.cvtColor(image_phone, cv2.COLOR_RGB2BGR)

            image_color = self.data.get_color_image(frame_index)
            image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)

            image_depth = self.data.get_depth_image(frame_index)
            depth_normalized = cv2.normalize(image_depth, None, 0, 255,
                                             cv2.NORM_MINMAX)
            depth_normalized = np.uint8(depth_normalized)
            image_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

            if stream or self.output_filename is not None:
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
                            f'Frame: {frame_index} / {frames}',
                            text_position, font, font_scale,
                            font_color, font_thickness)

                # Write to video if output path is set
                if self.output_filename and self.output_directory is not None:
                    output_video.write(combined)

            if stream:
                cv2.imshow('iPhone and RealSense Stream', combined)
                key = cv2.waitKey(self.interval_phone)

                if key & 0xFF == ord('q'):
                    break

            if frame_index % 100 == 0:
                print(f"Processed {frame_index} frames")

        if self.output_filename and self.output_directory is not None:
            output_video.release()
        cv2.destroyAllWindows()

# Demo


if __name__ == "__main__":
    # Change directory to the current directory
    # Necessary for running the script here in VS Code
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'data/')
    output_file = 'comparison_video_demo.mp4'

    # I want to double check file structure and data status
    # test = DataProcessor()
    # test.display_all_data()

    synchronizer = SensePhoneMerger(output_dir, output_file)
    synchronizer.perform_comparison(stream=True)