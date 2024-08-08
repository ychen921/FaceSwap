import cv2
import numpy as np

from Code.Mics.Utils import bilinearInterpolation

def tpsMatrix(x, y):
    x, y = x.reshape(1, -1), y.reshape(1, -1)
    
    z = np.zeros((x.shape[1], 1))
    k_x, k_y = x + z, y + z

    r = np.square((k_x - x.T)) + np.square((k_y - y.T))
    r[r == 0] = 1
    K = r * np.log(r)

    ones = np.ones((x.shape[1], 1))
    P = np.concatenate((x.T, y.T, ones), axis=1)
    
    upper_TPS = np.concatenate((K, P), axis=1)
    lower_TPS = np.concatenate((P.T, np.zeros((3,3))), axis=1)

    tpsMat = np.concatenate((upper_TPS, lower_TPS), axis=0)
    return tpsMat

def estimateParam(tpsMat, x, lambd):
    x = x.reshape((x.shape[0],1))
    x = np.concatenate((x, np.zeros((3,1))), axis=0)

    tpsMat = tpsMat + lambd * np.eye(tpsMat.shape[0])
    
    tspMat_inv = np.linalg.inv(tpsMat)
    spline = np.dot(tspMat_inv, x)

    return spline

def tpsWrap(dst_pts, src_x, src_y):
    dst_x = dst_pts[:,0].reshape(1, -1)
    dst_y = dst_pts[:,1].reshape(1, -1)
    
    r = np.square((dst_x - src_x)) + np.square((dst_y - src_y))
    r[r==0] = 1
    # y, x = np.where(r == 0)
    K = r * np.log(r)

    return K

def thisPlateSpine(image1, image2, points_1, points_2, hulls_2):

    # source and target coordinates
    src_x, src_y = points_1[:, 0], points_1[:, 1]
    dst_x, dst_y = points_2[:, 0], points_2[:, 1]

    lambd = 0.0000001

    tspMat = tpsMatrix(dst_x, dst_y)
    x_spline = estimateParam(tspMat, src_x, lambd)
    y_spline = estimateParam(tspMat, src_y, lambd)

    # Face mask for target image
    mask = np.zeros((image2.shape[0], image2.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, hulls_2, (255, 255, 255))

    Y, X = np.where(mask==255)
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    K = tpsWrap(points_2, X, Y)
    K = np.concatenate((K, X, Y, np.ones((X.shape[0],1))), axis=1)

    x_hat = np.dot(K, x_spline)
    y_hat = np.dot(K, y_spline)

    x_hat[x_hat < 0] = 0
    y_hat[y_hat < 0] = 0

    x_hat[x_hat > image1.shape[1]] = image1.shape[1] - 1
    y_hat[y_hat > image1.shape[0]] = image1.shape[0] - 1

    face_coord = np.concatenate((x_hat, y_hat), axis=1)

    rect = cv2.boundingRect(np.array(hulls_2))
    rect_center = (int(rect[0]+rect[2]/2), int(rect[1]+rect[3]/2))
    
    ####################
    # mask = np.zeros(image2.shape, dtype=np.uint8)
    # cv2.fillPoly(mask, [np.int32(hulls_2)], (255, 255, 255))
    # cv2.imwrite('../Output/Wraped_tps_mask.jpg', mask)
    ####################


    pixel_values = bilinearInterpolation(pts=face_coord.T, image=image1)
    img2_copy = image2.copy()

    image2[Y[:,0], X[:,0]] = pixel_values
    # mask[Y[:,0], X[:,0]] = pixel_values

    final_swap = cv2.seamlessClone(image2, img2_copy, mask, rect_center, cv2.NORMAL_CLONE)
    
    # cv2.imwrite('../Output/Wraped_tps.jpg', mask)
    
    return final_swap