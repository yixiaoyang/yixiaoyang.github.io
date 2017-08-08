---
author: leon
comments: true
date: 2017-08-02 00:19:00+00:00
layout: post
title: '[图像处理]基于Python Opencv的数独解析程序'
categories:
- 图像处理
tags:
- opencv
- 图像处理
---

# 基于Python Opencv的数独解析程序

主程序脚本，只是个调用

```python
from os import walk
from subprocess import call
import os

for (dirpath, dirnames, filenames) in walk("data/src"):
    for filename in filenames:
        fullpath = dirpath + "/" + filename
        outFileName = filename+"-out.png"
        outFullPath =  "./data/build/"+ outFileName

        pyfile1 = os.getcwd() + "/sudoku-preprocess.py"
        pyfile2 = os.getcwd() + "/sudoku-split.py"
        call(["python",pyfile1, fullpath, outFullPath])
        call(["python",pyfile2, outFullPath])
        print(fullpath)

```
数据目录

```
yixiaoyang@[/devel/git/github/OpencvTutorial/sudoku] % tree ./data
./data
├── build
│   ├── sudoku1.jpg-out.png
│   ├── sudoku1.jpg-out.png-contours.png
│   ├── sudoku3.jpg-out.png
│   ├── sudoku3.jpg-out.png-contours.png
│   ├── sudoku4.jpg-out.png
│   └── sudoku4.jpg-out.png-contours.png
├── cut
├── src
│   ├── sudoku1.jpg
│   ├── sudoku2.jpg
│   ├── sudoku3.jpg
│   └── sudoku4.jpg
├── testset
└── trainset
```

### 分析和分割数独区域
1. 平滑：消除部分噪声
2. 适当膨胀腐蚀：消除间隙
3. canny：得到轮廓
4. 提取轮廓后，拟合外接多边形，面积最大的那个矩形区域就认为是数独区域
5. 提取数独区域，仿射，拉伸后简单消除畸变将图形变“正”

![1.png](http://cdn4.snapgram.co/images/2017/08/08/1.png)

```python
#!/usr/bin/env python
# encoding: utf-8

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os

if len(sys.argv) < 3:
    print("[ERR] parameter error. try to run 'python xxxx.py src dst'")
    exit()
src_file = sys.argv[1]
dst_file = sys.argv[2]

class Config:
    debug = False
    src = "sudoku3.jpg"
    dst = "1-output.png"

    # 最大所方长宽值
    max_size = 1024

    min_area = 0.05
    min_contours = 8
    threshold_thresh = 110
    epsilon_start = 50
    epsilon_step = 10

    hough_rho = 100
    hough_theta = (1*np.pi)/180

'''
@return     [top-left, top-right, bottom-right, bottom-left]
'''
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def point_distance(a,b):
    return int(np.sqrt(np.sum(np.square(a - b))))

'''
@func 寻找面积最大的最大包络矩形区域
'''
def max_contour_idx(contours):
    max_area = 0
    max_idx = 0
    if len(contours) == 0:
        return -1
    for idx,c in enumerate(contours):
        if len(c) < Config.min_contours:
            continue
        curArea = cv2.contourArea(c)
        #print curArea
        if curArea > max_area:
            max_area = curArea
            max_idx = idx
    return max_idx

def image_resize(img):
    srcWidth, srcHeight, channels = image.shape
    max_size = max(srcWidth, srcHeight)
    if max_size < Config.max_size:
        return img

    resize_n = (max_size-Config.max_size/2)/Config.max_size +1
    return cv2.resize(img,(int(srcHeight/resize_n), int(srcWidth/resize_n)),interpolation=cv2.INTER_AREA )

if not os.path.exists(src_file) :
    print("[ERR] src file %s not exised!"%src_file)
    exit()

image = cv2.imread(src_file)
image = image_resize(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 高斯平滑，消除噪声
binary = cv2.GaussianBlur(gray,(5,5),0)

# adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) -> dst
#binary = cv2.adaptiveThreshold(binary, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=5, C=3)
#_, binary = cv2.threshold(binary, thresh=20, maxval=255, type=cv2.THRESH_BINARY)
#cv2.imwrite("1-threshold.png", binary, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

# canny提取轮廓
binary = cv2.Canny(binary, threshold1=0, threshold2=128, apertureSize = 3)
#cv2.imwrite("3-canny.png", binary, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

# 提取轮廓后，拟合外接多边形（矩形）
_,contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#print("len(contours)=%d"%(len(contours)))

if Config.debug:
    image_copy = image
    for idx,c in enumerate(contours):
        cv2.drawContours(image_copy, contours, idx, (255, 0, 255))
    #cv2.imwrite("contours.png", image_copy, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

max_idx = max_contour_idx(contours)

if max_idx >= 0:
    c = contours[max_idx]
    epsilon = Config.epsilon_start

    # 拟合外接最小矩形，拟合的最小长度为epsilon像素
    # Python: cv2.approxPolyDP(curve, epsilon, closed[, approxCurve]) → approxCurve
    # approxCurve – Result of the approximation. The type should match the type of the input curve.
    #               In case of C interface the approximated curve is stored in the memory storage
    #               and pointer to it is returned.
    # epsilon – Parameter specifying the approximation accuracy. This is the maximum distance between
    #           the original curve and its approximation.
    # closed – If true, the approximated curve is closed (its first and last vertices are connected).
    #          Otherwise, it is not closed.
    approx = cv2.approxPolyDP(c,epsilon,True)
    #print("max_contour_idx,epsilon,len(approx),len(c)=%d,%d,%d,%d"%(max_idx,epsilon,len(approx),len(c)))
    if (len(approx) == 4):
        #approx = approx[0:4,]
        approx = approx.reshape((4, 2))

        # 点重排序, [top-left, top-right, bottom-right, bottom-left]
        src_rect = order_points(approx)

        cv2.drawContours(image, c, -1, (0,255,255),1)
        cv2.line(image, (src_rect[0][0],src_rect[0][1]),(src_rect[1][0],src_rect[1][1]),color=(100,255,100))
        cv2.line(image, (src_rect[2][0],src_rect[2][1]),(src_rect[1][0],src_rect[1][1]),color=(100,255,100))
        cv2.line(image, (src_rect[2][0],src_rect[2][1]),(src_rect[3][0],src_rect[3][1]),color=(100,255,100))
        cv2.line(image, (src_rect[0][0],src_rect[0][1]),(src_rect[3][0],src_rect[3][1]),color=(100,255,100))

        # 获取最小矩形包络
        rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box = box.reshape(4,2)
        box = order_points(box)
        w,h = point_distance(box[0],box[1]), \
              point_distance(box[1],box[2])
        # 透视变换
        dst_rect = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]],
            dtype="float32")
        M = cv2.getPerspectiveTransform(src_rect, dst_rect)

        # 获取变换后的图像
        warped = cv2.warpPerspective(gray, M, (w, h))
        cv2.imwrite(dst_file, warped, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        exit()

```

### 分割数字区域和块区域
1. 膨胀腐蚀：消除间隙
2. 寻找外形轮廓，使用形态学梯度：膨胀图与腐蚀图之差，留下的就是轮廓。这一步跟canny的效果区别需要对比一下
3. 二值阈值：上一步得到的值已经可以比较好的消除非边缘区域以及噪声，取一个适当的阈值就可以把数字和方格轮廓二值化提取出来
4. 再次轮廓检测。利用轮廓tree尽可能分析出已有的数组grid。用面积、拟合矩形的长宽比例等条件筛选出需要的区域
5. 猜测计算数独grid的行数、列数、行宽、列宽、行线宽、列线宽等值。方法是在以筛选出的格子矩形区域中分析众数的行数、列数、行宽、列宽，将其作为猜测值，算出行列数
6. 分析出没有提取出来的格子（虚拟格子）。方法是根据其相邻的已分析出的方格坐标推测虚拟格的坐标和长宽直到所有格子分析出来
7. 根据所有格子的位置将其从图片中切割出来，简单处理下就可以用于OCR的机器学习识别数据


![2.png](http://cdn2.snapgram.co/imgs/2017/08/08/2.png)

```python
#!/usr/bin/env python
# encoding: utf-8

import cv2
import numpy as np
from scipy.ndimage import label
from scipy import stats
import statistics
import sys
import os

if len(sys.argv) < 2:
    print("[ERR] parameter error. try to run 'python xxxx.py src'")
    exit()
src_file = sys.argv[1]
src_files = src_file.split("/")
src_filename = src_files[-1]

class Config:
    debug = True

    src = "1-output3.png"
    dst = "2-output.png"

    # 最大所方长宽值
    max_size = 1024

    # 最少5x5=25个像素点
    min_area = 9
    epsilon_start = 50
    epsilon_step = 10

    # 块长宽比最大差值
    max_rectx = 0.25
    # 默认猜测格子线宽度4px
    guess_line_w = 4

def image_resize(img):
    srcWidth, srcHeight, channels = image.shape
    max_size = max(srcWidth, srcHeight)
    if max_size < Config.max_size:
        return img

    resize_n = (max_size-Config.max_size/2)/Config.max_size +1
    return cv2.resize(img,(int(srcHeight/resize_n), int(srcWidth/resize_n)),interpolation=cv2.INTER_AREA )

if not os.path.exists(src_file) :
    print("[ERR] src file %s not exised!"%src_file)
    exit()


image = cv2.imread(src_file)
image = image_resize(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 高斯平滑，消除噪声
binary = gray

#binary = cv2.medianBlur(gray,5)
#binary = cv2.GaussianBlur(gray,(5,5),0)
#cv2.imwrite("2-0-gaussian.png", binary, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

# 寻找外形轮廓，使用形态学梯度：膨胀图与腐蚀图之差，留下的就是轮廓
element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (1, 1))
binary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, element)
#cv2.imwrite("2-0-morphologyEx.png", binary)

_, binary = cv2.threshold(binary, thresh=15, maxval=255, type=cv2.THRESH_BINARY)
#cv2.imwrite("2-1-threshold.png", binary, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

kernel = np.ones((3, 3), np.uint8)
binary = cv2.dilate(binary, kernel, iterations=1)
#cv2.imwrite("2-1-dilate.png", binary, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

# canny提取轮廓
binary = cv2.Canny(binary, threshold1=0, threshold2=128, apertureSize = 3)
#cv2.imwrite("2-3-canny.png", binary, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

_, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#_, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#print  ("len(contours)=%d" % len(contours))

parent_null_count = 0
origin_copy = image
epsilon = Config.epsilon_start
areas = []
for idx, h in enumerate(hierarchy[0]):
    nxt, pre, child, parent = h
    if parent == -1:
        c = contours[idx]

        # 包含点集最小面积的矩形，，这个矩形是可以有偏转角度的，可以与图像的边界不平行, 输出是矩形的四个点坐标
        #rect = cv2.minAreaRect(c)
        #box = cv2.boxPoints(rect)
        x,y,w,h = cv2.boundingRect(c)
        box = [x,y],[x+w,y],[x+w,y+h],[x,y+h]
        box = np.int0(box)
        curArea = cv2.contourArea(box)

        distance1 = cv2.norm(box[0],box[1])
        distance2 = cv2.norm(box[1],box[2])

        if min(distance1,distance2) < 4:
            cv2.drawContours(origin_copy, c, -1, (0, 255, 0), 1)
            #print (parent_null_count, nxt, pre, child, parent, curArea)
        else:
            ratio = distance1/distance2
            if np.abs(ratio-1.0) < Config.max_rectx:
                parent_null_count += 1
                cv2.drawContours(origin_copy,[box],0,(255,0,255),1)
                #cv2.putText(origin_copy,"%d"%idx, (x+10,y+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                areas.append((idx,curArea,x,y,w,h))
            else:
                pass
                #cv2.drawContours(origin_copy,[box],0,(0, 255, 0), 1)
            #print (parent_null_count,idx, nxt, pre, child, parent, curArea,x)

        if False:
            cv2.imshow('contours',origin_copy)
            k =cv2.waitKey(0)
            if k == 27:
                break

#cv2.imwrite("2-4-contours.png", origin_copy, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

areas = np.array(areas)
mode_result = stats.mode(areas)
_, marea, mx, my, mw, mh = mode_result.mode[0]
#print(marea, mx, my, mw, mh)

min_area = marea*0.75
area_idx = np.where(areas[:,1] >= min_area)
# areas是已经找到的合格矩形的集合
areas = areas[area_idx]

def guess_grid_wh(areas, cur_col, another_col, area_col, cur_mVal, contour_wh):
    area_idx = np.where((areas[:,cur_col] == cur_mVal))
    select_areas = areas[area_idx]
    sort_idx = np.argsort(select_areas[:,3])
    select_areas = select_areas[sort_idx]

    contour_gaps = []
    for idx in range(len(select_areas)-1):
        contour_gaps.append(np.abs(select_areas[idx][another_col]-select_areas[idx+1][another_col]))
    mode_result = stats.mode(contour_gaps)
    if(mode_result.mode > 0):
        mode_gap = mode_result.mode[0]
        # 得到的gap可能是相邻的两个块，有可能是中间间隔了n个的块，因此需要先推算这两个块间隔了多少个块
        steps = (int)(mode_gap/(contour_wh+Config.guess_line_w)+0.5)
    return (mode_gap/steps+1)

guess_grid_h = guess_grid_wh(areas, 2, 3, 1, mx, mh)
guess_grid_w = guess_grid_wh(areas, 3, 2, 1, my, mw)

# rows, columns and channels
srcRows, srcCols, channels = image.shape
guess_grid_cols, guess_grid_rows =  (int)(srcCols/guess_grid_w+0.5), (int)(srcRows/guess_grid_h+0.5)
guess_line_xw = int((srcCols - guess_grid_w*guess_grid_cols)/guess_grid_cols)+1
guess_line_yw = int((srcRows - guess_grid_h*guess_grid_rows)/guess_grid_rows)+1
print("file:%s"%(src_file), "cols, rows, w, h=",guess_grid_cols, guess_grid_rows, guess_line_xw, guess_line_yw)


# 开始分区
anchor_x_diff = guess_grid_w*0.4
anchor_y_diff = guess_grid_h*0.4
content_areas = {}
for row in range(guess_grid_rows):
    anchor_y = int(guess_grid_h*row)
    for col in range(guess_grid_cols):
        anchor_x = int(guess_grid_w*col)
        # 查找在(anchor_x, anchor_y)附近的块，如果没有，则构造一个虚拟块, _, marea, mx, my, mw, mh
        area_idx = np.where((np.abs(areas[:,2]-anchor_x) < anchor_x_diff) & (np.abs(areas[:,3]-anchor_y) < anchor_y_diff))
        area = areas[area_idx]
        #print(area)
        if len(area) > 0:
            content_areas[row,col] = area[0]
            #print(row, col, area[0])
#print(content_areas)


# 构造虚拟块
for row in range(guess_grid_rows):
    for col in range(guess_grid_cols):
        rect = 0,0,0,0
        if (row,col) in content_areas:
            area = content_areas[row,col]
            rect = int(area[2]), int(area[3]), int(area[4]), int(area[5])
            #print(row, col, area)
        else:
            guess_x, guess_y, guess_w, guess_h, x_cnt, y_cnt = 0, 0, 0, 0, 0, 0

            guess_x = col * guess_grid_w
            guess_y = row * guess_grid_h
            guess_w = int(col*guess_grid_w)-1
            guess_h = int(row*guess_grid_h)-1

            # 位置图：
            #       1
            #       |
            #   2 - O - 4
            #       |
            #       8
            position = 0
            if (row,col-1) in content_areas:
                position += 2
            if (row,col+1) in content_areas:
                position += 4
            if (row-1,col) in content_areas:
                position += 1
            if (row+1,col) in content_areas:
                position += 8

            # 0    1     2  3  4  5
            # idx, area, x, y, w, h
            area1 = content_areas.get((row-1,col))
            area2 = content_areas.get((row,col-1))
            area4 = content_areas.get((row,col+1))
            area8 = content_areas.get((row+1,col))

            if position == 0:
                pass
            if position == 1:
                guess_x = area1[2]
                guess_y = area1[3] + guess_grid_h + guess_line_yw
                guess_w = area1[4]
                guess_h = area1[5]
            if position == 2:
                guess_x = area2[2] + guess_grid_w + guess_line_xw
                guess_y = area2[3]
                guess_w = area2[4]
                guess_h = area2[5]
            if position == 3:
                guess_x = area1[2]
                guess_y = area2[3]
                guess_w = int(statistics.mean([area2[4],area1[4]]))
                guess_h = int(statistics.mean([area2[5],area1[5]]))
            if position == 4:
                guess_x = area4[2] - guess_grid_w - guess_line_xw
                guess_y = area4[3]
                guess_w = area4[4]
                guess_h = area4[5]
            if position == 5:
                guess_x = area1[2]
                guess_y = area4[3]
                guess_w = int(statistics.mean([area4[4],area1[4]]))
                guess_h = int(statistics.mean([area4[5],area1[5]]))
            if position == 6:
                guess_x = int(statistics.mean([area2[2],area4[2]]))
                guess_y = int(statistics.mean([area2[3],area4[3]]))
                guess_w = int(statistics.mean([area2[4],area4[4]]))
                guess_h = int(statistics.mean([area2[5],area4[5]]))
            if position == 7:
                guess_x = int(statistics.mean([area1[2],area2[2],area4[2]]))
                guess_y = int(statistics.mean([area2[3],area4[3]]))
                guess_w = int(statistics.mean([area1[4],area2[4],area4[4]]))
                guess_h = int(statistics.mean([area1[5],area2[5],area4[5]]))
            if position == 8:
                guess_x = area8[2]
                guess_y = area8[3] - guess_grid_h - guess_line_yw
                guess_w = area8[4]
                guess_h = area8[5]
            if position == 9:
                guess_x = int(statistics.mean([area1[2],area8[2]]))
                guess_y = int(statistics.mean([area1[3],area8[3]]))
                guess_w = int(statistics.mean([area1[4],area8[4]]))
                guess_h = int(statistics.mean([area1[5],area8[5]]))
            if position == 10:
                guess_x = area8[2]
                guess_y = area2[3]
                guess_w = int(statistics.mean([area8[4],area2[4]]))
                guess_h = int(statistics.mean([area8[5],area2[5]]))
            if position == 11:
                guess_x = int(statistics.mean([area1[2],area8[2]]))
                guess_y = int(statistics.mean([area1[3],area2[3],area8[3]]))
                guess_w = int(statistics.mean([area1[4],area2[4],area8[4]]))
                guess_h = int(statistics.mean([area1[5],area2[5],area8[5]]))
            if position == 12:
                guess_x = area8[2]
                guess_y = area4[3]
                guess_w = int(statistics.mean([area8[4],area4[4]]))
                guess_h = int(statistics.mean([area8[5],area4[5]]))
            if position == 13:
                guess_x = int(statistics.mean([area1[2],area8[2]]))
                guess_y = int(statistics.mean([area1[3],area4[3],area8[3]]))
                guess_w = int(statistics.mean([area1[4],area4[4],area8[4]]))
                guess_h = int(statistics.mean([area1[5],area4[5],area8[5]]))
            if position == 14:
                guess_x = int(statistics.mean([area8[2],area2[2],area4[2]]))
                guess_y = int(statistics.mean([area2[3],area4[3]]))
                guess_w = int(statistics.mean([area8[4],area2[4],area4[4]]))
                guess_h = int(statistics.mean([area8[5],area2[5],area4[5]]))
            if position == 15:
                guess_x = int(statistics.mean([area1[2],area2[2],area4[2],area8[2]]))
                guess_y = int(statistics.mean([area1[3],area2[3],area4[3],area8[3]]))
                guess_w = int(statistics.mean([area1[4],area2[4],area4[4],area8[4]]))
                guess_h = int(statistics.mean([area1[5],area2[5],area4[5],area8[5]]))

            rect = int(guess_x), int(guess_y), int(guess_w), int(guess_h)
            content_areas[row,col] = (0, guess_w*guess_h, guess_x, guess_y, guess_w, guess_h)

            pt1 = (int(guess_x), int(guess_y))
            pt2 = (int(guess_x+guess_w), int(guess_y+guess_h))
            cv2.rectangle(origin_copy,pt1,pt2,(256,128,0),1)

        x,y,w,h = rect
        cut_img = gray[y:y+h, x:x+w]
        cut_img = cv2.resize(cut_img,(32,32))
        #_, cut_img = cv2.threshold(cut_img, thresh=64, maxval=255, type=cv2.THRESH_BINARY)
        cut_img = cv2.adaptiveThreshold(cut_img, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=5, C=3)
        cv2.imwrite("./data/cut/"+src_filename+"-%d-%d.png"%(row,col), cut_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

cv2.imwrite("./data/build/"+src_filename+"-contours.png", origin_copy, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

```

### 数字识别

（待完成）

### 解数独

（待完成）
