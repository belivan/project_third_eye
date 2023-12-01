import pyrealsense2 as rs
import cv2
import numpy as np
import os


def _extract_sense_bag(bag_file, save_dir, max_frames=667):
    # File path to the .bag file and save directory
    os.makedirs(save_dir, exist_ok=True)

    # Configure the pipeline to stream from the .bag file
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file)
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)

    # Start streaming from the file
    profile = pipeline.start(config)

    # Create an align object
    align = rs.align(rs.stream.color)

    frame_count = 0

    depth_frames = np.empty((max_frames, 720, 1280), dtype=np.uint16)
    color_frames = np.empty((max_frames, 720, 1280, 3), dtype=np.uint8)

    while frame_count < max_frames:
        frame_count += 1

        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

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

        depth_frames[frame_count - 1] = depth_image
        color_frames[frame_count - 1] = color_image

        text = f"Extracting Intel Real Sense Frame: {frame_count}/{max_frames}"
        print(text)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

        if frame_count >= max_frames:
            break

    pipeline.stop()
    cv2.destroyAllWindows()

    np.save(os.path.join(save_dir, "color_frames.npy"), color_frames)
    np.save(os.path.join(save_dir, "depth_frames.npy"), depth_frames)

    return True


def _extract_phone(video):
    cap = cv2.VideoCapture(video)  # open video
    frames = []  # create empty list
    while True:
        ret, frame = cap.read()  # read video
        if not ret:
            break
        frames.append(frame)   # save the frame
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
