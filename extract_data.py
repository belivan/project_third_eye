import pyrealsense2 as rs
import cv2
import numpy as np
import os


def _extract_sense_bag(bag_file, save_dir):
    # File path to the .bag file and save directory
    os.makedirs(save_dir, exist_ok=True)

    # Configure the pipeline to stream from the .bag file
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file, repeat_playback=False)
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)

    # Start streaming from the file
    profile = pipeline.start(config)

    # Create an align object
    align = rs.align(rs.stream.color)

    frame_count = 0

    depth_frames = []
    color_frames = []

    while True:
        # Get frameset of color and depth
        try:
            frames = pipeline.wait_for_frames()
        except RuntimeError as e:
            print(f"An error occurred while waiting for frames (not an error): {e}")
            break  # No more frames available

        if not frames:
            break  # No more frames available

        frame_count += 1

        try:
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

        except RuntimeError as e:
            print(f"An error occurred while processing frame {frame_count}: {e}")
            continue

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.array(color_frame.get_data())
        depth_image = np.array(depth_frame.get_data())

        depth_frames.append(depth_image)
        color_frames.append(color_image)

        text = f"Extracting Intel Real Sense Frame: {frame_count}"
        print(text)

    pipeline.stop()
    cv2.destroyAllWindows()

    np.save(os.path.join(save_dir, "color_frames.npy"), color_frames)
    np.save(os.path.join(save_dir, "depth_frames.npy"), depth_frames)

    return True


def _extract_phone(video):
    cap = cv2.VideoCapture(video)  # open video
    frames = []  # create empty list
    frame_count = 0
    while True:
        frame_count += 1
        ret, frame = cap.read()  # read video
        if not ret:
            break
        frames.append(frame)   # save the frame
        print(f"Extracting Phone Frame: {frame_count}")
    frames = np.array(frames)  # convert everything to numpy array
    np.save("data/phone_frames.npy", frames)  # save the numpy array
    return True


def extract_data(bag_file, video_file):
    """
    Extracts the data from the bag file and the video file and saves them as NumPy arrays.
    """

    color_file = os.path.join("data/", "color_frames.npy")
    depth_file = os.path.join("data/", "depth_frames.npy")
    phone_file = os.path.join("data/", "phone_frames.npy")

    if (not os.path.isfile(color_file) or not os.path.isfile(depth_file)) and _extract_sense_bag(bag_file, "data/"):
        print("Successfully extracted data from the bag file")
    else:
        print("Data already extracted from the bag file")

    # Extract data from the video file
    if not os.path.isfile(phone_file) and _extract_phone(video_file):
        print("Successfully extracted data from the video file")
    else:
        print("Data already extracted from the video file")

    return color_file, depth_file, phone_file
