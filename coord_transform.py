import cv2 as cv
import numpy as np

'''
Below function takes an array of 3D coordinates and converts them into pixel coordinates
You need to enter intrinsic and extrinsic camera matrices
If the io matrix is different you can change that as well but wont be necessary in.
'''

def coord2point(coordinates, R, t, A, io=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])):
    pixels = []

    # Calculate Rt and P outside the loop
    Rt = np.concatenate((R, t), axis=1)
    Rt = np.concatenate((Rt, np.array([[0, 0, 0, 1]])), axis=0)
    P = A @ io @ Rt
    
    for coordinate in coordinates:
        if coordinate is None or len(coordinate) == 0:
            continue       
        homoCoordinate = np.concatenate([coordinate, np.array([1])])
        m = P @ homoCoordinate
        image_coord = m[:2] / m[2]
        image_coord = image_coord.astype(int).reshape([2, ])
        pixels.append(image_coord)

    return pixels

# Test the function
def main():
    oImage = cv.imread("origin_0.5x.png")
    h, w = oImage.shape[:2]

    # Extrinsic
    # Rotation matrix from camera to origin
    R = np.array([[0.0, 0.0, -1.0],
                  [0.0, 1.0, 0.0],
                  [1.0, 0.0, 0.0]])

    # Translation matrix from camera to the global origin

    t = np.array([[4.2672 * np.cos(np.pi / 4),
                   0.0,

                   4.2672 * np.cos(np.pi / 4) + 0.860]]).T
    # Intrinsic

    f = 0.00277  # meters
    delta = 1.4e-6  # meters
    Cu = w / 2
    Cv = h / 2

    A = np.array([[f / delta, 0.0, Cu],
                  [0.0, f / delta, Cv],
                  [0.0, 0.0, 1.0]])

    # -----------------------main code---------------------------

    coords = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    pixels = coord2point(coords, R, t, A)

    for coord in pixels:
        cv.circle(oImage, coord, 10, (0, 0, 255), -1)  # draw circle at the point

    cv.imshow("oImage", oImage)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
