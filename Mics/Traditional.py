import cv2
import numpy as np

from Mics.delaunayTriangulation import delaunayTriangulation, srcTraingulation
from Mics.ThinPlateSpline import thisPlateSpine
from Mics.Utils import bilinearInterpolation

def InternalCoordinates(Barycentric, xy_coord):
    sum = np.sum(Barycentric, axis=0)
    sum = np.round(sum, 4)
    sum_greater = sum > 0
    sum_less = sum <= 1
    sum = sum_greater * sum_less

    alpha = Barycentric[0, :]
    alpha_greater = alpha >= -0.000001
    alpha_less = alpha <= 1
    alpha = alpha_greater * alpha_less

    beta = Barycentric[1, :]
    beta_greater = beta >= -0.000001
    beta_less = beta <= 1
    beta = beta_greater * beta_less

    gamma = Barycentric[2, :]
    gamma_greater = gamma >= -0.000001
    gamma_less = gamma <= 1
    gamma = gamma_greater * gamma_less

    internal_index = alpha * beta * gamma * sum
    internal_pts = Barycentric[:, internal_index]
    dst_internal_pts = xy_coord[:, internal_index]

    return internal_pts, dst_internal_pts

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

        dst_corners = dst_tri[i]
        # compose of B matrix for computing Barycentric coordinate
        B = np.array([[dst_corners[0], dst_corners[2], dst_corners[4]],
                      [dst_corners[1], dst_corners[3], dst_corners[5]],
                      [             1,              1,              1]])
        B_inv = np.linalg.inv(B)
        
        rect = boundingRect(dst_corners)

        x = np.arange(rect[0]-1, rect[2]+1)
        y = np.arange(rect[1]-1, rect[3]+1)
        x_mesh = np.repeat(x, y.shape[0])
        y_mesh = np.tile(y, x.shape[0])
        x_mesh = x_mesh.reshape((1, x_mesh.shape[0]))
        y_mesh = y_mesh.reshape((1, y_mesh.shape[0]))
        xy_coor = np.concatenate((x_mesh, y_mesh, np.ones((1, x_mesh.shape[1]))), axis=0)

        Barycentric = np.dot(B_inv, xy_coor)
        internal_pts, img2_interal_pts = InternalCoordinates(Barycentric, xy_coor)
        img2_interal_pts = img2_interal_pts.astype(np.int64)

        # Compute the corresponding pixel position in the source image
        src_corners = src_tri[i]
        A = np.array([[src_corners[0], src_corners[2], src_corners[4]],
                      [src_corners[1], src_corners[3], src_corners[5]],
                      [              1,               1,               1]])
        src_coord = np.dot(A, internal_pts)

        # Convert the values to homogeneous coordinates
        src_coord = src_coord / src_coord[2]

        # Bilinear Interpolation
        pixel_values = bilinearInterpolation(pts=src_coord, image=image1)
        image2[img2_interal_pts[1], img2_interal_pts[0]] = pixel_values
    
    # Blending
    rect = cv2.boundingRect(hull_2)
    rect_center = (int(rect[0]+rect[2]/2), int(rect[1]+rect[3]/2))
    mask = np.zeros(image2.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(hull_2)], (255, 255, 255))

    final_swap = cv2.seamlessClone(np.uint8(image2), np.uint8(img2_copy), mask, rect_center, cv2.NORMAL_CLONE)
    return final_swap

def triangulation(image1, image2, points_1, points_2, hull2, method):

    # Delanauy traingulation for convex hull points
    rect = (0, 0, image2.shape[1], image2.shape[0])

    lm_points_2 = []
    for p in points_2:
        lm_points_2.append((int(p[0]), int(p[1])))

    dst_triangles = delaunayTriangulation(image2, rect, lm_points_2)
    src_traingles = srcTraingulation(image1, dst_triangles, points_1, lm_points_2,)

    final_swap = triangulationWrap(image1, image2, src_tri=src_traingles, dst_tri=dst_triangles, hull_2=np.array(hull2))
    return final_swap
    

def traditionalFaceSwap(image_1, image_2, points_1, points_2, hull_2, Method, show=False):
    if (Method == "tps"):
        final_swap = thisPlateSpine(image_1, image_2, np.array(points_1), np.array(points_2), hull_2)
    else:
        final_swap = triangulation(image_1, image_2, points_1, points_2, hull_2, Method)

    if show:
        cv2.imshow("Face Swap - tradition", final_swap)
        cv2.waitKey(1)