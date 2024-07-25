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
    elif point[0] > rectangle[0] + rectangle[2]:
        return False
    elif point[1] > rectangle[1] + rectangle[3]:
        return False
    return True

def delaunayTriangulation(img, rect, points):
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
        subdiv.insert(tuple(map(int, p)))

    triangleList = subdiv.getTriangleList()
    
    delaunayTri = []
    pt = []

    for t in triangleList:

        # Three points of a triangle
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = tuple(map(int, (t[0], t[1])))
        pt2 = tuple(map(int, (t[2], t[3])))
        pt3 = tuple(map(int, (t[4], t[5])))

        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            ind = []
            for i in range(3):
                for j in range(len(points)):
                    if (abs(pt[i][0] - points[j][0]) < 1.0 and abs(pt[i][1] - points[j][1]) < 1.0):
                        ind.append(j)
            if (len(ind) == 3):
                delaunayTri.append((ind[0], ind[1], ind[2]))

            # Draw delaunay triangle
            cv2.line(img_copy, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA, 0)
            cv2.line(img_copy, pt2, pt3, (255, 255, 255), 1, cv2.LINE_AA, 0)
            cv2.line(img_copy, pt3, pt1, (255, 255, 255), 1, cv2.LINE_AA, 0)

        pt = []
        
    return delaunayTri
    # cv2.imshow("win_delaunay", img_copy)
    # cv2.waitKey(100)