import cv2
import numpy as np

def rect_contains(rectangle, point):
    """
    Check if a point is inside a rectangle
    """
    if point[0] < rectangle[0]:
        return False
    elif point[1] < rectangle[1]:
        return False
    elif point[0] > rectangle[2]:
        return False
    elif point[1] > rectangle[3]:
        return False
    return True

def delaunayTriangulation(img, rect, points, show=True):
    """
    Calculate Dalaunay Traingulation
    refer to https://learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/
    """
    # Keep a copy around
    img_copy = img.copy()

    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect)
    
    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    triangleList = subdiv.getTriangleList()

    if show == True:
        for t in triangleList:
            pt1 = tuple(map(int, (t[0], t[1])))
            pt2 = tuple(map(int, (t[2], t[3])))
            pt3 = tuple(map(int, (t[4], t[5])))

            if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
                # Draw delaunay triangle
                cv2.line(img_copy, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA, 0)
                cv2.line(img_copy, pt2, pt3, (255, 255, 255), 1, cv2.LINE_AA, 0)
                cv2.line(img_copy, pt3, pt1, (255, 255, 255), 1, cv2.LINE_AA, 0)

        for p in points:
            cv2.circle(img_copy, p, 1, (0,0,255), 2)

        # cv2.imshow("win_delaunay", img_copy)
        cv2.imwrite('../Output/delaunay_tri.jpg', img_copy)
        # cv2.waitKey(1)

    return triangleList

def srcTraingulation(image1, dst_triangles, points_1, points_2, show=False):
    image_copy = image1.copy()
    
    src_triangles = []
    for i in range(len(dst_triangles)):
        ind = []

        pt1 = (dst_triangles[i][0], dst_triangles[i][1])
        pt2 = (dst_triangles[i][2], dst_triangles[i][3])
        pt3 = (dst_triangles[i][4], dst_triangles[i][5])
        
        ind.append(points_2.index(pt1))
        ind.append(points_2.index(pt2))
        ind.append(points_2.index(pt3))
        move = [points_1[ind[0]][0], points_1[ind[0]][1], \
                points_1[ind[1]][0], points_1[ind[1]][1], \
                points_1[ind[2]][0], points_1[ind[2]][1]]
        src_triangles.append(move)

    if show:
        for t in src_triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            cv2.line(image_copy, pt1, pt2, (255,255,255), 1)
            cv2.line(image_copy, pt2, pt3, (255,255,255), 1)
            cv2.line(image_copy, pt1, pt3, (255,255,255), 1)
        cv2.imshow("Source Frame", image_copy)
        cv2.waitKey(1)

    return np.asarray(src_triangles)