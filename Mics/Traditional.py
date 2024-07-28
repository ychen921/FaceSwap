import cv2
import numpy as np

from Mics.delaunayTriangulation import delaunayTriangulation, srcTraingulation



def boundingRect(points):
    rect = list()
    x = np.array([points[0], points[2], points[4]])
    y = np.array([points[1], points[3], points[5]])

    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    rect = [xmin, ymin, xmax, ymax]

    return rect

def triangulationWrap(image1, image2, src_tri, dst_tri, hull_2):
    img2_copy = image2.copy()

    for i in range(len(dst_tri)):
        rect = boundingRect(corners)

        corners = dst_tri[i]
        # compose of B matrix for computing Barycentric coordinate
        B = np.array([[corners[0], corners[2], corners[4]],
                      [corners[1], corners[3], corners[5]],
                      [         1,          1,          1]])
        B_inv = np.linalg.inv(B)

        x = np.arange(rect[0]-1, rect[2]+1)
        y = np.arange(rect[1]-1, rect[3]+1)
        x_mesh = np.repeat(x, y.shape[0])
        y_mesh = np.tile(y, x.shape[0])
        x_mesh = x_mesh.reshape((1, x_mesh.shape[0]))
        y_mesh = y_mesh.reshape((1, y_mesh.shape[0]))
        x_ycoor = np.concatenate((x_mesh, y_mesh, np.ones((1, x_mesh.shape[1]))), axis=0)

        Barycentric = np.dot(B_inv, x_ycoor)


def triangulation(image1, image2, points_1, points_2, hull2, method):

    # Delanauy traingulation for convex hull points
    rect = (0, 0, image2.shape[1], image2.shape[0])

    lm_points_2 = []
    for p in points_2:
        lm_points_2.append((int(p[0]), int(p[1])))

    dst_triangles = delaunayTriangulation(image2, rect, lm_points_2)
    src_traingles = srcTraingulation(image1, dst_triangles, points_1, lm_points_2,)


def traditionalFaceSwap(image_1, image_2, points_1, points_2, hull_2, Method):
    if (Method == "tps"):
        pass
    else:
        triangulation(image_1, image_2, points_1, points_2, hull_2, Method)