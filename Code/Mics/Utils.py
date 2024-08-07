import cv2
import numpy as np


def bilinearInterpolation(pts, image):
    pixel_values = np.zeros((pts.shape[1], 3))
    xy_coor = pts[0:2, :]

    up_x = np.ceil(pts[0,:]).astype(np.uint64)
    up_y = np.ceil(pts[1,:]).astype(np.uint64)

    up_x[up_x>=image.shape[1]] = image.shape[1] - 1
    up_y[up_y>=image.shape[0]] = image.shape[0] - 1

    down_x = np.floor(pts[0,:]).astype(np.uint64)
    down_y = np.floor(pts[1,:]).astype(np.uint64)

    a = xy_coor[0,:] - down_x
    b = xy_coor[1,:] - down_y

    wt_top_right = (a*b).reshape((pts.shape[1],1))
    wt_top_left = ((1-a)*b).reshape((pts.shape[1],1))
    wt_down_left = ((1-a)*(1-b)).reshape((pts.shape[1],1))
    wt_down_right = (a*(1-b)).reshape((pts.shape[1],1))

    wt_top_right = np.repeat(wt_top_right, 3, axis=1)
    wt_top_left = np.repeat(wt_top_left, 3, axis=1)
    wt_down_left = np.repeat(wt_down_left, 3, axis=1)
    wt_down_right = np.repeat(wt_down_right, 3, axis=1)

    pixel_values = (wt_top_right*image[up_y[:],up_x[:]]) + (wt_top_left*image[up_y[:],down_x[:]]) + \
            (wt_down_left*image[down_y[:],down_x[:]]) + (wt_down_right*image[down_y[:],up_x[:]])
    
    pixel_values[pixel_values>255] = 255

    return pixel_values.astype(np.uint8)