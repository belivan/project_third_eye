import cv2
import cv2 as cv
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import torch


def person_mask(image):
    model = YOLO("yolov8n-seg.pt")  # choose the YOLO model you want to use
    results = model.predict(image)  # feed the image into the model
    result = results[0]  # extract the result 0 since we only fed one image to the model

    for box in result.boxes:  # result.boxes will contain the objects detected

        if box.cls == 0:  # if the box is a person
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # return np.array([x1, y1, x2, y2], dtype=int)

    mask = result.masks
    keypoints = result.keypoints

    contours = np.array(mask.xy, dtype=int)

    blank = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    cv.drawContours(blank, contours, -1, 255, cv.FILLED)

    x, y = np.where(blank == 255)

    return np.array([x, y]).T


if __name__ == "__main__":
    color_image = cv.imread("data/intel pictures/color_frame_0007.png")
    depth_image = np.load("data/intel pictures/depth_frame_0007.npy")
    mask = person_mask(color_image)
    cv.imshow("image", color_image)


    cv.waitKey(0)
    cv.destroyAllWindows()
