import os
import cv2
import numpy as np
import argparse

from Code.Phase1.FacialLandmarks import facialLandmarksDetection, twoFaceLandmarkDetection
from Code.Phase1.Traditional import traditionalFaceSwap

from Code.Phase2.api_ import PRN_
from Code.Phase2.api import PRN
from Code.Phase2.prnet import *


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', default="../Data/TestSet_P2/", type=str,help='Directory to the input data.')
    Parser.add_argument('--VideoName', default='Test4', type=str,help='File name of the input video stream.')
    Parser.add_argument('--ImageName', default='Rambo', type=str,help='File name of the target the face image.')
    Parser.add_argument('--Mode', type=int, default=1, help='Mode 1 for swapping the face in video with an image. Mode 2 for swap two faces within a video')
    Parser.add_argument('--Method', default='tps', type=str, help='tri, tps, prnet')
    Parser.add_argument('--SaveFileName', default='FaceSwap', type=str,help='Name of face swap saved video')

    Args = Parser.parse_args()
    DataPath = Args.DataPath
    VideoName = Args.VideoName
    ImageName = Args.ImageName
    SaveFileName = Args.SaveFileName
    Mode = Args.Mode
    Method = Args.Method

    # Testing data path
    VideoPath = DataPath + VideoName + '.mp4'
    ImagePath = DataPath + ImageName + '.jpg'

    # Video save path
    if not os.path.exists("../Output"):
        os.makedirs("../Output")
    save_path = "../Output/" + VideoName + "-" + "mode" + str(Mode) + "-" + Method + ".avi"

    # Create video capture object
    cap = cv2.VideoCapture(VideoPath)
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4))
    clip_save = cv2.VideoWriter(save_path,  
                            cv2.VideoWriter_fourcc(*'DIVX'), 
                            10, (frame_width, frame_height))

    # Count the number of frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    VideoDuration = totalNoFrames // fps

    print('Video duration in seconds: ', VideoDuration, 's')
    print('Number of frames of the video: ', totalNoFrames)

    # Capture video frame
    ret, frame = cap.read()
        
    if (Mode == 1):
        if (Method.lower() == "prnet"):
            prn = PRN(is_dlib = True)
        print('----------- Mode 1: face swap using an image ------------')
        print('Tradition method: ', Method)

        # Read source face image
        face_image = cv2.imread(ImagePath)
        # face_image = cv2.resize(face_image, (int(0.8*face_image.shape[1]), int(0.8*face_image.shape[0])), interpolation=cv2.INTER_LINEAR)

        # Detect facial landmarks and return number of 
        # face points and their coordinate
        _, points_1, _ = facialLandmarksDetection(face_image)
        
        count = 1
        prev_pos = None
        while(cap.isOpened()):
            ret, frame = cap.read()
            count+=1
            
            if (ret == True):

                # Face swap using Position map Regression Network (PRNet)
                if (Method.lower() == "prnet"):
                    pos = prn.process(frame)
                    ref_pos = prn.process(face_image)

                    if pos is None:
                        if prev_pos is not None:
                            pos = prev_pos
                        else:
                            print('Frame no.', count, ' -- no faces detected')
                            final_wrap = frame

                    if pos is not None:
                        print('Frame no.', count)
                        final_wrap = FaceSwap_DL(prn, pos, ref_pos, frame, face_image)

                # Traditional method (face wraping using TPS or triangulation)
                else:
                    # Detect facial landmarks on video frame
                    num_face, points_2, hull_2 = facialLandmarksDetection(frame)

                    # if no faces detected, continue
                    if (num_face == 0):
                        print('Frame no.', count, ' -- no faces detected')
                        final_wrap = frame

                    else:
                        print('Frame no.', count)
                        # Implement traditional method of face swap
                        final_wrap = traditionalFaceSwap(image_1=face_image, image_2=frame, points_1=points_1, points_2=points_2, hull_2 = hull_2, Method=Method, show=False)

                    
                cv2.imshow("Face swapped: Mode 1", final_wrap)
                clip_save.write(final_wrap)
                cv2.waitKey(10)

                if cv2.waitKey(1) & 0xff==ord('q'):
                    cv2.destroyAllWindows()
                    break

            else:
                break
            

    # Swap 2 faces in a video
    else :
        if (Method.lower() == "prnet"):
            prn = PRN_(is_dlib = True)
        print('----------- Mode 2: swap 2 faces in a video ------------')
        print('Tradition method: ', Method)
        count = 1
        # prev_poses = None
        while(cap.isOpened()):
            # Capture video frame
            ret, frame = cap.read()
            count+=1
            
            if (ret == True):
                # Face swap using Position map Regression Network (PRNet)
                if (Method.lower() == "prnet"):

                    poses = prn.process(frame)
                    if poses is None:
                        poses = prev_poses

                    # if len(poses) < 2:
                    #     poses = prev_poses

                    if len(poses) == 2:
                        print('Frame no.', count)
                        prev_poses = poses
                        pose1 ,pose2 = poses[0],poses[1]
                        temp = FaceSwap_DL(prn, pose1, pose2, frame, frame)
                        final_wrap = FaceSwap_DL(prn, pose2, pose1, temp, frame)

                    else:
                        print("number of Faces found...", len(poses))
                        final_wrap = frame

                # Traditional method (TPS or Triangulation)
                else:
                    # Detect facial landmarks on video frame
                    num_face, points_list, hull_list = twoFaceLandmarkDetection(frame)

                    # if no faces detected, continue
                    if num_face != 2:
                        print('Frame no.', count, ' -- Detect less then 2 faces')
                        final_wrap = frame

                    else:
                        print('Frame no. ', count)

                        # Implement traditional method of face swap
                        points_1, points_2 = points_list

                        hull_1 = []
                        hull_1.append(hull_list[:2][0])

                        hull_2 = []
                        hull_2.append(hull_list[:2][1])
                        frame_copy = frame.copy()

                        temp = traditionalFaceSwap(frame, frame, points_2, points_1, hull_1, Method)
                        final_wrap = traditionalFaceSwap(frame_copy, temp, points_1, points_2, hull_2, Method)
                
                cv2.imshow("Face swapped: Mode 2", final_wrap)
                clip_save.write(final_wrap)
                cv2.waitKey(10)

                if cv2.waitKey(1) & 0xff==ord('q'):
                    cv2.destroyAllWindows()
                    break
            
            else:
                break
        
        cap.release()
        clip_save.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()