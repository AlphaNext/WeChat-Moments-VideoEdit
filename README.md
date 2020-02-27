# WeChat-Moments-VideoEdit
GIF format Emoji add and tracking on WeChat Moments Video Edit Stage

## 1. Prerequisites
    * OpenCV-3.x
    * ImageMagick (library for read GIF)

## 2. How to run
    * mode 1: just run and show processed frame
    ```
    make
    ./main tree.mp4 LOL_kasha.gif 
    ```
    * mode 2: run and show processed frame, and save result
    
    ```
    # remove commit in line 27: #define SAVE_VIDEO 1
    make
    ./main tree.mp4 LOL_kasha.gif demo.mp4
    ```
    
## 3. WeChat-Moments-Edit-Demo (:point_left: vs :point_right:) My-Edit-Demo
<p align="middle">
  <img src="vs_show/wechat_demo.gif" width="400" />
  <img src="vs_show/my_demo.gif" width="400" /> 
</p>

## 4. Python version coming soon ......

