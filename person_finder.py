import cv2
import cv2 as cv
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import torch


class person_detection():
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self._load_model(model_name)

    def _load_model(self, model_name):
        self.model = YOLO(model_name)

    def get_box(self, image):
        results = self.model.predict(image)  # feed the image into the model
        result = results[0]  # extract the result 0 since we only fed one image to the model

        for box in result.boxes:  # result.boxes will contain the objects detected

            if box.cls == 0:  # if the box is a person
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                return np.array([x1, y1, x2, y2], dtype=int)

    def person_mask(self, image):
        results = self.model.predict(image)  # feed the image into the model
        result = results[0]  # extract the result 0 since we only fed one image to the model

        mask = result.masks
        keypoints = result.keypoints

        contours = np.array(mask.xy, dtype=int)

        blank = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        cv.drawContours(blank, contours, -1, 255, cv.FILLED)

        x, y = np.where(blank == 255)

        return np.array([x, y]).T

    def get_keypoints(self, image):
        results = self.model.predict(image)  # feed the image into the model
        result = results[0]  # extract the result 0 since we only fed one image to the model

        keypoints = result.keypoints.xy[0].tolist()

        return np.array(keypoints, dtype=int)


if __name__ == "__main__":
    pd = person_detection("yolov8n-pose.pt")
    color_image = cv.imread("data2/intel pictures/color_frame_0007.png")
    depth_image = np.load("data2/intel pictures/depth_frame_0007.npy")
    keypoints = pd.get_keypoints(color_image)

    for point in keypoints:
        cv.circle(color_image, point, 3, (0, 0, 255), -1)
    cv.imshow("image", color_image)

    cv.waitKey(0)
    cv.destroyAllWindows()
