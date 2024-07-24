import os
import cv2
import numpy as np
import argparse

from Mics.FacialLandmarks import facialLandmarksDetection


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', default="./Data/TestSet_P2/", type=str,help='Directory to the input data.')
    Parser.add_argument('--VideoName', default='Test1', type=str,help='File name of the input video stream.')
    Parser.add_argument('--ImageName', default='Rambo', type=str,help='File name of the target the face image.')
    Parser.add_argument('--Mode', type=int, default=1, help='Mode 1 for swapping the face in video with an image. Mode 2 for swap two faces within a video')
    Parser.add_argument('--method', default='TPS', type=str, help='affine, tri, tps, prnet')

    Args = Parser.parse_args()
    DataPath = Args.DataPath
    VideoName = Args.VideoName
    ImageName = Args.ImageName
    Mode = Args.Mode

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

    # Capture video frame's height and width
    ret, frame = cap.read()
    v_height, v_width = frame.shape[0], frame.shape[1]

    if (Mode == 1):
        face_image = cv2.imread(ImagePath)

        # Detect facial landmarks
        num_points, points_image = facialLandmarksDetection(face_image)
    else :
        pass

if __name__ == "__main__":
    main()