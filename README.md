# Faster RCNN, Physics Based CT scans Augmentation

This project is designed to investigate the impact of physcis-based augmentation within a faster RCNN network to detect and localize abnormalities that a radiologist may identify during CT scan reviews.

---

## Detection algorithm

The detection algorithm is built upon Faster RCNN [[1]](#1) and enhanced using the physics-based augmentation [[2]](#2), accounting for the noise effect s caused by different reconstruction kernels.

---

## Reconstruction kernels and noise texture
Following noise textures illustrates examples of reconstruction kernels used for noise insertion simulation.

![Alt Text](/Figures/Kernels.gif)

## Requirenment
Python 3.10

## Instruction



## References
<a id="1">[1]</a> 
Shaoqing R. (2015). 
Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems 28 (2015).

<a id="2">[2]</a> 
https://github.com/DIDSR/DICaugment
