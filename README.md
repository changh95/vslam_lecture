# Code exercises for FastCampus VSLAM for Augmented Reality lecture series

This repository contains code exercises for the SLAM section in the lecture series - '[Computer Vision Signature](https://fastcampus.co.kr/data_online_cvsignature)' at FastCampus. This lecture series is delivered in Korean language.

![](title.png)

## How to use

Most of the code exercises are based on the base docker image. The base docker image contains numerous C++ libraries for SLAM, such as OpenCV, Eigen, Sophus and GTSAM. 

You can build the base docker image using the following command. 

```shell
docker build . --tag slam:latest --progress=plain
echo "xhost +local:docker" >> ~/.profile
```

## Table of contents

- Chapter 1: Introduction to lecture series
  - 1.1 Lecture introduction
  - 1.2 How to properly study SLAM
  - 1.3 [Basic C++ / CMake](1_3)
- Chapter 2: VSLAM basics
  - 2.1 What is VSLAM?
  - 2.2 3D rotation and translation
  - 2.3 Homogeneous coordinates
  - 2.4 [Code exercises for 3D rotation and translation](2_4)
  - 2.5 Camera models
  - 2.6 IMU
  - 2.7 [System calibration for VIO](2_7)
- Chapter 3: Image processing and multiple-view geometry
  - 3.1 Local feature extraction & matching
  - 3.2 [Local feature extraction & matching, using OpenCV library](3_2)
  - 3.3 Global feature extraction
  - 3.4 [Bag of Visual Words, using DBoW2 library](3_4)
  - 3.5 Feature tracking
  - 3.6 Advanced feature tracking
  - 3.7 Epipolar geometry & Homography
  - 3.8 [Tutorial for Essential / Fundamental / Homography matrix](3_8)
  - 3.9 Triangulation & PnP
  - 3.10 [Tutorial for Triangulation & PnP](3_10)
  - 3.11 RANSAC
  - 3.12 [Tutorial for RANSAC](3_12)
- Chapter 4: Graph SLAM
  - 4.1 Graph SLAM
  - 4.2 Bundle adjustment
  - 4.3 [Tutorial for Bundle adjustment](4_3)
- Final projects:
  - [Basalt VIO for augmented reality](basalt)