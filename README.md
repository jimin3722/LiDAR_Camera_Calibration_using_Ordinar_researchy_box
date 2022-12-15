# LiDAR_Camera_Calibration_using_Ordinary_box

Introduction
------------
This project implements the extrinsic calibration method of the ``Accurate Calibration of LiDAR-Camera Systems Using Ordinary Boxes``. I tested the LiDAR - Camera system. 
* LiDAR : ``Velodyne VLP-16`` 
* RGB Camera : ``Intel RealSense D435``
* Official [paper](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w6/html/Pusztai_Accurate_Calibration_of_ICCV_2017_paper.html).

Result
------

This repo follows under step.
<figure>
  <img src="./asset/images/step.png" height='75%' width ='75%'>
</figure>

1. Search for plane candidates and find outliers
<figure>
  <img src="./asset/images/find_planes.png" height='30%' width ='30%'><img src="./asset/images/remove_outlier.png" height='27.7%' width ='27.7%'>
</figure>

2. Select 3 planes and box fitting
<figure>
  <img src="./asset/gif/double_filter.gif" height='42%' width ='42%'>
</figure>

3. Box refinement
<figure>
  <img src="./asset/gif/case1.gif" height='45%' width ='45%'><img src="./asset/gif/case2.gif" height='45%' width ='45%'>
</figure>

4. Reprojection
<figure>
  <img src="./asset/images/box_reprojection.png" height='50%' width ='50%'>
</figure>


Dependencies
------------
It is tested with opencv-4.2.0 
Visualization used Matplotlib3D and Open3D.

This repo reused 3d ransac code from ``pyranscas3d/plane.py`` [code](https://github.com/leomariga/pyRANSAC-3D/blob/master/pyransac3d/plane.py)


Citation
-------- 
```
Mariga, L. (2022). pyRANSAC-3D (Version v0.6.0) [Computer software]. https://doi.org/10.5281/zenodo.7212567

@InProceedings{Pusztai_2017_ICCV,
    author = {Pusztai, Zoltan and Hajder, Levente},
    title = {Accurate Calibration of LiDAR-Camera Systems Using Ordinary Boxes},
    booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV) Workshops},
    month = {Oct},
    year = {2017}
}
```
