# FaceSwap
The aim of this project is to implement an end-to-end pipline to swap faces in a video just like [Snapchat's face swap filter](https://www.google.com). It is a fairly complicated procedure and variants of the approach that implemented in many movies.

## Phase 1: Traditional Approach
| ![](./Figures/Overview.png)| 
|:--:| 
| *Overview of the face replacement pipeline.* |

### 1. Facial landmarks Detection

The first step in the traditional approach is to find facial landmarks. One of the major reasons to use facial landmarks instead of using all points on the face is to reduce the computational complexity. However, better results can be obtained using all points (dense flow). For facial landmarks detection, we utilized 68-point facial landmarks detector in dlib library that built into OpenCV.

The figure below shows the result of detected 68-points facial landmarks marked as red points and a green bounding box around the face.

| ![](./Figures/FacialLandmarks.jpg)| 
|:--:| 
| *Output of dlib for facial landmarks detection.* |

### 2. Face Wrapping
After we obtained facial landmarks, we need to wrap faces in 3D and one such method is obtained by drawing the dual of the Voronoi diagram, i.e., connecting each two neighboring sites in the Voronoi diagram. This is called the Delaunay Triangulation which tries the maximize the smallest angle in each triangle. The figure below shows the result of Delaunay Triangulation of the target face image. Next, we will wrap the face using 2 methods: **Triangulation** and **Thin Plate Spline**.

| ![](./Figures/delaunay_tri.jpg)| 
|:--:| 
| *Delaunay Triangulation on target face we want to swap* |

#### 2.1. Face Wrapping using Triangulation
Since, Delaunay Triangulation tries the maximize the smallest angle in each triangle, we obtained the same triangulation in both the images (source and target face images). If we have correspondences between the facial landmarks, we also have correspondences between the triangles.

Now we need to wrap the target face to the source face (we are using inverse warping). The reason using forward wraping is since the pixels in the source image are transfered in the target image by computing their new coordinates; this often leads to a non uniform distribution of "known pixels" in the target image, making it hard to reconstruct. The inverse warping aproach is based on finding for every pixel in the target image the corresponding pixel from the source image. This will lead to a better target image at least by exposing the reconstruction problems.

To implement triangulation wrapping, we follow these steps:

1. For each triangle in the target face B, compute the Barycentric coordinate.
2. Compute the corresponding pixel position in the source image A using the barycentric equation but different triangle coordinates.
3. Copy back the value of the pixel at (xA,yA) to the target location.

#### 2.2. Face Wrapping using Thin Plate Spline