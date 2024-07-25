import cv2
import numpy as np

from Mics.delaunayTriangulation import delaunayTriangulation

def triangulation(image1, image2, image1Wraped, hull1, hull2, method):

    # Delanauy traingulation for convex hull points
    rect = (0, 0, image2.shape[1], image2.shape[0])
    dt = delaunayTriangulation(image2, rect, hull2)

    if len(dt) == 0:
        quit()

def traditionalFaceSwap(image_1, image_2, points_1, points_2, Method):
    hull1, hull2 = [], []
    img1Warped = np.copy(image_2)
    hullIndex = cv2.convexHull(np.array(points_2), returnPoints=False)
    
    for i in range(len(hullIndex)):
        hull1.append(points_1[int(hullIndex[i])])
        hull2.append(points_2[int(hullIndex[i])])

    if (Method == "tps"):
        pass
    else:
        triangulation(image_1, image_2, img1Warped, hull1, hull2, Method)