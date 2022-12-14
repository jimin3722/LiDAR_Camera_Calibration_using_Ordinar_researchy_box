# LiDAR_Camera_Calibration_using_Ordinary_box

Introduction
------------
This project implements the extrinsic calibration method of the ``Accurate Calibration of LiDAR-Camera Systems Using Ordinary Boxes``. I tested the LiDAR - Camera system. 
* LiDAR : ``Velodyne VLP-16`` 
* RGB Camera : ``Intel RealSense D435``
* Official [paper](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w6/html/Pusztai_Accurate_Calibration_of_ICCV_2017_paper.html).

Result
------
Left is a official image in paper. Right is my experiment. 

Dependencies
------------
It is tested with opencv-4.1.15. 
Visualization used Matplotlib3D and Open3D.

Citation
-------- 
```
@InProceedings{Pusztai_2017_ICCV,
    author = {Pusztai, Zoltan and Hajder, Levente},
    title = {Accurate Calibration of LiDAR-Camera Systems Using Ordinary Boxes},
    booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV) Workshops},
    month = {Oct},
    year = {2017}
}
```
