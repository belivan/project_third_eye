import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import requests
import os
import cv2


# import imageio  # for creating gifs

# models can be movenet_lightning_f16, movenet_thunder_f16, movenet_lightning_int8, movenet_thunder_int8

class PoseEstimator:
    KEYPOINT_EDGE_INDS_TO_COLOR = {
        (0, 1): 'm',
        (0, 2): 'c',
        (1, 3): 'm',
        (2, 4): 'c',
        (0, 5): 'm',
        (0, 6): 'c',
        (5, 7): 'm',
        (7, 9): 'm',
        (6, 8): 'c',
        (8, 10): 'c',
        (5, 6): 'y',
        (5, 11): 'm',
        (6, 12): 'c',
        (11, 12): 'y',
        (11, 13): 'm',
        (13, 15): 'm',
        (12, 14): 'c',
        (14, 16): 'c'
    }

    def __init__(self, model_name):
        self.model_name = model_name
        self.input_size = None
        self.interpreter = None
        self.module = None
        self._initialize_model()

    def _initialize_model(self):
        """Initializes the model based on its type (TFLite or TensorFlow Hub)."""
        tflite_models = ["movenet_lightning_f16", "movenet_thunder_f16",
                         "movenet_lightning_int8", "movenet_thunder_int8"]

        hub_models = ["movenet_lightning", "movenet_thunder"]

        if self.model_name in tflite_models:
            model_path = f"{self.model_name}.tflite"
            self._load_tflite_model(model_path)

        elif self.model_name in hub_models:
            self._load_hub_model()
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

    def _load_tflite_model(self, model_path):
        """Loads a TFLite model, downloading it if necessary."""
        if not os.path.exists(model_path):
            print("Downloading from internet")
            self._download_tflite_model(model_path)

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        # Set the input size based on the model

        if "lightning" in self.model_name:
            self.input_size = 192
        elif "thunder" in self.model_name:
            self.input_size = 256
        else:
            raise ValueError("Unsupported TFLite model for input size setting.")

    def _load_hub_model(self):
        """Loads a model from TensorFlow Hub."""
        urls = {
            "movenet_lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
            "movenet_thunder": "https://tfhub.dev/google/movenet/singlepose/thunder/4"
        }
        if self.model_name in urls:
            self.module = hub.load(urls[self.model_name])
            self.input_size = 192 if "lightning" in self.model_name else 256
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

    def _download_tflite_model(self, model_path):
        """Downloads the TFLite model from TensorFlow Hub."""
        url = self._get_model_url()
        response = requests.get(url)
        if response.status_code != 200:
            raise ConnectionError("Failed to download the model.")
        with open(model_path, "wb") as file:
            file.write(response.content)

    def _get_model_url(self):
        """Returns the URL of the TFLite model based on the model name."""
        urls = {
            "movenet_lightning_f16": "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite",
            "movenet_thunder_f16": "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite",
            "movenet_lightning_int8": "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite",
            "movenet_thunder_int8": "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite"
        }
        if self.model_name in urls:
            return urls[self.model_name]
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

    def _movenet(self, input_image):
        """
        Runs the pose estimation model on the input image.

        Args:
            input_image: A [1, height, width, 3] tensor representing the input image.

        Returns:
            keypoints_with_scores: The predicted keypoints with scores.
        """
        if self.interpreter:
            return self._run_tflite_inference(input_image)
        elif self.module:
            return self._run_hub_inference(input_image)
        else:
            raise RuntimeError("Model is not loaded properly.")

    def _run_tflite_inference(self, input_image):
        """
        Runs inference using the TFLite model.

        Args:
            input_image: A [1, height, width, 3] tensor representing the input image.

        Returns:
            Predicted keypoint coordinates and scores.
        """
        input_image = tf.cast(input_image, dtype=tf.uint8)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]['index'], input_image.numpy())

        self.interpreter.invoke()

        keypoints_with_scores = self.interpreter.get_tensor(output_details[0]['index'])
        return keypoints_with_scores

    def _run_hub_inference(self, input_image):
        """
        Runs inference using the TensorFlow Hub model.

        Args:
            input_image: A [1, height, width, 3] tensor representing the input image.

        Returns:
            Predicted keypoint coordinates and scores.
        """
        model = self.module.signatures['serving_default']
        input_image = tf.cast(input_image, dtype=tf.int32)

        outputs = model(input_image)
        keypoints_with_scores = outputs['output_0'].numpy()
        return keypoints_with_scores

    def _keypoints_and_edges_for_display(self, keypoints_with_score, height, width, keypoint_threshold=0.11):
        """
        Processes keypoints with scores to determine which keypoints are above
        a certain confidence threshold, and prepares them for display.

        Args:
            keypoints_with_score (ndarray): Array of keypoints with scores.
            height (int): Height of the image.
            width (int): Width of the image.
            keypoint_threshold (float, optional): Threshold for keypoint confidence. Defaults to 0.11.

        Returns:
            ndarray: Array of high confidence keypoints.
            ndarray: Array of edges connecting keypoints.
            list: List of colors for each edge.
        """
        # Initialize lists to store keypoints, edges and their colors
        keypoints_all = []
        keypoint_edges_all = []
        edge_colors = []

        # Determine the number of instances (people) in the input
        num_instances, _, _, _ = keypoints_with_score.shape

        # Iterate over each instance
        for id in range(num_instances):
            # Extract x, y coordinates and scores of keypoints for the instance
            kpts_x = keypoints_with_score[0, id, :, 1]
            kpts_y = keypoints_with_score[0, id, :, 0]
            kpts_scores = keypoints_with_score[0, id, :, 2]

            # Calculate absolute x and y coordinates based on image dimensions
            kpts_abs_xy = np.stack([width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)

            # Filter keypoints based on confidence threshold
            kpts_above_thrs_abs = kpts_abs_xy[kpts_scores > keypoint_threshold, :]
            keypoints_all.append(kpts_above_thrs_abs)

            # Determine edges (connections between keypoints) for keypoints above threshold
            for edge_pair, color in self.KEYPOINT_EDGE_INDS_TO_COLOR.items():
                if (kpts_scores[edge_pair[0]] > keypoint_threshold and
                        kpts_scores[edge_pair[1]] > keypoint_threshold):
                    # Start and end coordinates of the edge
                    x_start = kpts_abs_xy[edge_pair[0], 0]
                    y_start = kpts_abs_xy[edge_pair[0], 1]
                    x_end = kpts_abs_xy[edge_pair[1], 0]
                    y_end = kpts_abs_xy[edge_pair[1], 1]
                    line_seg = np.array([[x_start, y_start], [x_end, y_end]])

                    # Add the edge and its color to the respective lists
                    keypoint_edges_all.append(line_seg)
                    edge_colors.append(color)

        # Combine all keypoints and edges for display
        keypoints_xy = np.concatenate(keypoints_all, axis=0) if keypoints_all else np.zeros((0, 17, 2))
        edges_xy = np.stack(keypoint_edges_all, axis=0) if keypoint_edges_all else np.zeros((0, 2, 2))

        return keypoints_xy, edges_xy, edge_colors

    def predict_keypoints_transform(self, image):
        """
        Predicts the keypoints and transforms the image.

        Args:
            image: A [height, width, 3] tensor or numpy array representing an RGB image.

        Returns:
            original_image: The original image as a numpy array.
            keypoints: A list of keypoints in the format [(x1, y1), (x2, y2), ...].
        """
        # aspect_ratio = float(width) / height

        # add one more dimension to an array [100, 100, 3] -> [1, 100, 100, 3] (neccessary in tensorflow)
        input_image = tf.expand_dims(image, axis=0)
        # resize and pad the image
        input_image = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)

        # Run model prediction
        keypoints_with_scores = self._movenet(input_image)

        # Visualize the predictions with image
        display_image = tf.cast(tf.image.resize_with_pad(input_image, 1280, 1280), dtype=tf.int32)
        height, width, _ = (np.squeeze(display_image.numpy(), axis=0)).shape

        # Get keypoints
        (keypoints, _, _) = self._keypoints_and_edges_for_display(keypoints_with_scores, height, width)

        # Remove Image Padding and adjust keypoints
        padded_image = np.squeeze(display_image.numpy(), axis=0)
        padded_height, padded_width, _ = padded_image.shape

        # Calculate padding thickness
        top_padding_thickness = (padded_height - image.shape[0]) // 2
        left_padding_thickness = (padded_width - image.shape[1]) // 2

        # Adjust keypoints based on padding thickness
        adjusted_keypoints = [[x - left_padding_thickness, y - top_padding_thickness] for x, y in keypoints]

        # Round the coordinates to get integer indices
        adjusted_keypoints = np.round(adjusted_keypoints).astype(int)

        return adjusted_keypoints


# Example usage
if __name__ == "__main__":
    pose_estimator = PoseEstimator("movenet_thunder_f16")

    # Single Image Example
    image_path = 'data/intel pictures/color_frame_0050.png'
    image = cv2.imread(image_path)
    keypoints = pose_estimator.predict_keypoints_transform(image)

    for pose_keypoint in keypoints:
        cv2.circle(image, pose_keypoint, 5, (255, 0, 0), -1)

    cv2.imshow("test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
