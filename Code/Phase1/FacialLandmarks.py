import cv2
import numpy as np
import dlib
from imutils import face_utils

def facialLandmarksDetection(image):
    """
    Detect face landmarks
    Input: Image(np array)
    Output: Number of landmarks (int)
            landmark coordinates (list)
    """
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./Code/Mics/shape_predictor_68_face_landmarks.dat')
    
    faces = detector(gray_image, 1)
    
    face_coordinates = []
    hull_list = []
    for i, face in enumerate(faces):
        shape = predictor(gray_image, face)
        shape = face_utils.shape_to_np(shape)

        # Extract face rectangle coordinate and size
        (x, y, w, h) = face_utils.rect_to_bb(face)

        # Draw a green rectangle around the face
        cv2.rectangle(RGB_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw circles on the facial landmarks
        for (x, y) in shape:
            cv2.circle(RGB_image, (x, y), 5, (0, 0, 255), -1)
            face_coordinates.append((x, y))

        hull = cv2.convexHull(np.array(face_coordinates), False)
        hull = hull.reshape((hull.shape[0], hull.shape[2]))
        hull_list.append(hull)

    return len(faces), face_coordinates, hull_list

def twoFaceLandmarkDetection(image):
    """
    Detect face landmarks
    Input: Image(np array)
    Output: Number of landmarks (int)
            landmark coordinates (list)
    """

    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./Code/Mics/shape_predictor_68_face_landmarks.dat')
    
    faces = detector(gray_image, 1)

    face_coordinates = []
    coordinate_list = []
    hull_list = []

    if (len(faces) == 2):
        for i, face in enumerate(faces):
            shape = predictor(gray_image, face)
            shape = face_utils.shape_to_np(shape)

            # Extract face rectangle coordinate and size
            (x, y, w, h) = face_utils.rect_to_bb(face)

            # Draw a green rectangle around the face
            cv2.rectangle(RGB_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Draw circles on the facial landmarks
            for (x, y) in shape:
                cv2.circle(RGB_image, (x, y), 5, (0, 0, 255), -1)
                face_coordinates.append((x, y))

            hull = cv2.convexHull(np.array(face_coordinates), False)
            hull = hull.reshape((hull.shape[0], hull.shape[2]))
            hull_list.append(hull)
            
            coordinate_list.append(face_coordinates)
            face_coordinates = []

    return len(faces), coordinate_list, hull_list