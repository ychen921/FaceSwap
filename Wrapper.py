import os
import cv2
import numpy as np
import argparse

from Mics.FacialLandmarks import facialLandmarksDetection, twoFaceLandmarkDetection
from Mics.Traditional import traditionalFaceSwap


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', default="./Data/TestSet_P2/", type=str,help='Directory to the input data.')
    Parser.add_argument('--VideoName', default='Test2', type=str,help='File name of the input video stream.')
    Parser.add_argument('--ImageName', default='Steven', type=str,help='File name of the target the face image.')
    Parser.add_argument('--Mode', type=int, default=1, help='Mode 1 for swapping the face in video with an image. Mode 2 for swap two faces within a video')
    Parser.add_argument('--Method', default='tps', type=str, help='affine, tri, tps, prnet')

    Args = Parser.parse_args()
    DataPath = Args.DataPath
    VideoName = Args.VideoName
    ImageName = Args.ImageName
    Mode = Args.Mode
    Method = Args.Method

    VideoPath = DataPath + VideoName + '.mp4'
    ImagePath = DataPath + ImageName + '.jpg'

    # Create video capture object
    cap = cv2.VideoCapture(VideoPath)

    # Count the number of frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    VideoDuration = totalNoFrames // fps

    print('Video duration in seconds: ', VideoDuration, 's')
    print('Number of frames of the video: ', totalNoFrames)

    # Capture video frame
    ret, frame = cap.read()


    if (Mode == 1):
        print('----------- Mode 1: face swap using an image ------------')
        print('Tradition method: ', Method)

        # Read source face image
        face_image = cv2.imread(ImagePath)
        # face_image = cv2.resize(face_image, (int(0.8*face_image.shape[1]), int(0.8*face_image.shape[0])), interpolation=cv2.INTER_LINEAR)

        # Detect facial landmarks and return number of 
        # face points and their coordinate
        _, points_1, _ = facialLandmarksDetection(face_image)

        count = 1
        while(cap.isOpened()):
            ret, frame = cap.read()
            count+=1
            
            if (ret == True):

                # Face swap using Position map Regression Network (PRNet)
                if (Method.lower() == "prnet"):
                    pass

                # Traditional method (face wraping using TPS or triangulation)
                else:
                    # Detect facial landmarks on video frame
                    num_face, points_2, hull_2 = facialLandmarksDetection(frame)

                    # if no faces detected, continue
                    if (num_face == 0):
                        print('Frame no.', count, ' -- no faces detected')
                        # continue

                    else:
                        print('Frame no.', count)
                        # Implement traditional method of face swap
                        frame = traditionalFaceSwap(image_1=face_image, image_2=frame, points_1=points_1, points_2=points_2, hull_2 = hull_2, Method=Method, show=False)

                    
                cv2.imshow("Face swapped: Mode 1", frame)
                cv2.waitKey(10)

                if cv2.waitKey(1) & 0xff==ord('q'):
                    cv2.destroyAllWindows()
                    break

            else:
                exit()
            

    # Swap 2 faces in a video
    else :
        print('----------- Mode 2: swap 2 faces in a video ------------')
        print('Tradition method: ', Method)
        count = 1
        while(cap.isOpened()):
            # Capture video frame
            ret, frame = cap.read()
            count+=1
            
            if (ret == True):
                # Face swap using Position map Regression Network (PRNet)
                if (Method.lower() == "prnet"):
                    pass

                # Traditional method (TPS or Triangulation)
                else:
                    # Detect facial landmarks on video frame
                    num_face, points_list, hull_list = twoFaceLandmarkDetection(frame)

                    # if no faces detected, continue
                    if num_face != 2:
                        print('Frame no.', count, ' -- Detect less then 2 faces')

                    else:
                        print('Frame no. ', count)

                        # Implement traditional method of face swap
                        points_1, points_2 = points_list

                        hull_1 = []
                        hull_1.append(hull_list[:2][0])

                        hull_2 = []
                        hull_2.append(hull_list[:2][1])

                        temp = traditionalFaceSwap(frame, frame, points_1, points_2, hull_2, Method)
                        frame = traditionalFaceSwap(frame, temp, points_2, points_1, hull_1, Method)
                
                cv2.imshow("Face swapped: Mode 2", frame)
                cv2.waitKey(10)

                if cv2.waitKey(1) & 0xff==ord('q'):
                    cv2.destroyAllWindows()
                    break
            
            else:
                exit()


if __name__ == "__main__":
    main()