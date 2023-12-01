import cv2
import os
import numpy as np


class VideoSynchronizer:
    def __init__(self, folder1, folder2, output_filename, total_duration):
        self.folder1 = folder1
        self.folder2 = folder2
        self.output_filename = output_filename
        self.images1 = np.load(folder1)  # intel
        self.images2 = np.load(folder2)  # origin
        
        self.total_duration = len(self.images2)/30.0  # total_duration
        print("Total duration: ", self.total_duration)
        self.fps2 = float(30.0)
        
        self.interval1 = int((self.total_duration / len(self.images1)) * 1000)
        self.fps1 = float(1000/self.interval1)
        print("FPS depth: ", self.fps1)

    def map_frame(self, folder1_frame_index):
        """
        Maps frame index from folder1 to folder2.
        """
        ratio = len(self.images2) / len(self.images1)
        return int(folder1_frame_index * ratio)

    def setup_video_writer(self):
        """
        Sets up a video writer.
        """
        example_img1 = cv2.imread(self.images1[0])
        example_img2 = cv2.imread(self.images2[0])

        if example_img2.shape != example_img1.shape:
            example_img2 = cv2.resize(example_img2, (example_img1.shape[1], example_img1.shape[0]))
        combined_example = np.hstack((example_img1, example_img2))
        height, width, _ = combined_example.shape

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        return cv2.VideoWriter(self.output_filename, fourcc, 1000 / self.interval1, (width, height))

    def process_videos(self, video=True, stream=True):
        if stream:
            cv2.namedWindow('Comparison', cv2.WINDOW_NORMAL)
        save_dir = 'data/origin_sync/'
        os.makedirs(save_dir, exist_ok=True)

        if video:
            output_video = self.setup_video_writer()  # video writer

        current_frame = 0

        phone_frames = np.empty((len(self.images2), 720, 1280, 3), dtype=np.uint8)

        # break_flag = False
        # while not break_flag:
        for index1 in range(len(self.images1)):
            current_frame += 1
            
            image1 = self.images1[index1]
            # path1 = os.path.join(self.folder1, img1)
            # image1 = cv2.imread(path1)

            index2 = self.map_frame(index1)

            print("Index 1: ", index1, " Index 2: ", index2)

            if index2 < len(self.images2):
                image2 = self.images2[index2]
                # path2 = os.path.join(self.folder2, img2)
                # image2 = cv2.imread(path2)

                # frame_filename = os.path.join(save_dir, f'frame_{current_frame:04d}.png')
                # cv2.imwrite(frame_filename, image2)
                phone_frames[current_frame - 1] = image2

            else:
                image2 = np.zeros_like(image1)

            if stream or video:
                if image2.shape != image1.shape:
                    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

                combined = np.hstack((image1, image2))
                if video:
                    output_video.write(combined)  # video writer

            if stream:
                cv2.imshow('Comparison', combined)
                key = cv2.waitKey(self.interval1)

                if key & 0xFF == ord('q'):
                    # break_flag = True
                    break

        print(f"Frames saved of origin: {current_frame}/{len(self.images2)}")
        if video:
            output_video.release()  # video writer
        cv2.destroyAllWindows()

        np.save(os.path.join(save_dir, 'phone_frames.npy'), phone_frames)


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)  # Gets the directory where the script is located
    # Change the current directory to the directory where the bag file is located
    # if running from VSCode, uncomment the following line
    os.chdir(current_dir)

    synchronizer = VideoSynchronizer('data/color_frames.npy', 'data/phone_frames.npy', 'comparison_video.avi', 26)
    synchronizer.process_videos(video=False, stream=False)