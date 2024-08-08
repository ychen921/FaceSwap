# FaceSwap
The aim of this project is to implement an end-to-end pipline to swap faces in a video just like [Snapchat's face swap filter](https://www.google.com). We implemented two mode. One is swap a source face image to a video also with a face. Another one will be swapping two faces in a video.

## Phase 1: Traditional Approach
| ![](./Figures/Overview.png)| 
|:--:| 
| *Overview of the face replacement pipeline.* |

### 1. Facial landmarks Detection

The first step in the traditional approach is to find facial landmarks. One of the major reasons to use facial landmarks instead of using all points on the face is to reduce the computational complexity. However, better results can be obtained using all points (dense flow). For facial landmarks detection, we utilized 68-point facial landmarks detector in `dlib` library that built into OpenCV.

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
Since, Delaunay Triangulation tries the maximize the smallest angle in each triangle, we obtained the same triangulation in both the images (source and target face images). If we have correspondences between the facial landmarks, we also have correspondences between the triangles. We can make the assumption that in each triangle the content is planar (forms a plane in 3D) and hence the warping between the the triangles in two images is affine.

Now we need to wrap the target face to the source face (we are using inverse warping). The reason using forward wraping is since the pixels in the source image are transfered in the target image by computing their new coordinates; this often leads to a non uniform distribution of "known pixels" in the target image, making it hard to reconstruct. The inverse warping aproach is based on finding for every pixel in the target image the corresponding pixel from the source image. This will lead to a better target image at least by exposing the reconstruction problems.

To implement triangulation wrapping, we follow these steps:

1. For each triangle in the target face B, compute the Barycentric coordinate.
2. Compute the corresponding pixel position in the source image A using the barycentric equation but different triangle coordinates.
3. Copy back the value of the pixel at (xA,yA) to the target location.

#### 2.2. Face Wrapping using Thin Plate Spline
Another way to do the transformation is by using Thin Plate Splines (TPS) which can model arbitrarily complex shapes. We want to compute a TPS that mpas form the target feature points in image B to the corresponding source feature points in image A. Note that we need two splines, one for the x coordinate and one for the y. A thin plate spline has the following form:

$$
f(x,y) = a_1 + (a_x)x + (a_y)y + \sum_{i=1}^p{w_i U\left( \vert \vert (x_i,y_i) - (x,y)\vert \vert_1\right)}
$$

Here, $$ U(r) = r^2\log (r^2 ) $$

Note that, again in this case we are performing inverse warping. Warping using a TPS is performed in two following steps:

1. We will estimate the parameters of the TPS. The solution of the TPS model rquires solving the following equation:

$$
 \begin{bmatrix} K & P\\ P^T & 0\\ \end{bmatrix} 
  \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_p \\ a_x \\ a_y \\ a_1  \end{bmatrix}  =
  \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_p \\ 0 \\ 0 \\ 0 \end{bmatrix}  
$$

where \\( K_{ij} = U\left( \vert \vert (x_i,y_i)-(x_j,y_j) \vert \vert_1 \right)\\). $$v_i = f(x_i,y_i)$$ and the i<sup>th</sup> row of $$P$$ is $$(x_i, y_i, 1)$$. $$K$$ is a matrix of size size $$p \times p$$, and $$P$$ is a matrix of size $$p \times 3$$. In order to have a stable solution you need to compute the solution by: