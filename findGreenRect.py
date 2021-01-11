import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math


hsv_min = np.array((85, 28, 65), np.uint8)
hsv_max = np.array((140, 95, 66), np.uint8)
color_blue = (255,0,0)
color_yellow = (0,255,255)

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result


def update(inimg):
    
    vis = inimg.copy()
    cv.imshow('pcb', vis)

def unwarp(img, src, dst, testing):
    h, w = img.shape[:2]
    # use cv.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv.getPerspectiveTransform(src, dst)
    # use cv.warpPerspective() to warp your image to a top-down view
    warped = cv.warpPerspective(img, M, (w, h), flags=cv.INTER_LINEAR)

    if testing:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        f.subplots_adjust(hspace=.2, wspace=.05)
        ax1.imshow(img)
        x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
        y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
        ax1.plot(x, y, color='red', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
        ax1.set_ylim([h, 0])
        ax1.set_xlim([0, w])
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(cv.flip(warped, 1))
        ax2.set_title('Unwarped Image', fontsize=30)
        plt.show()
    else:
        return warped, M

def getDest(ww,hh,rot=False):
    if rot == True:
        tmp=ww
        ww=hh
        hh=tmp


    dst = np.float32([(0, 0),
                      (0, ww),
                      (hh, ww),
                      (hh, 0)]) 
    return dst, ww, hh  

def findContours(imgin,size, DrawInfo=False):
    # преобразуем в hsv палитру
    hsv = cv.cvtColor( imgin, cv.COLOR_BGR2HSV )
    # цвет платы на белом
    thresh= cv.inRange(hsv, (25, 25, 25), (75, 255,255))
    # ищем контуры
    contours0, hierarchy = cv.findContours( thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # ищем самый большой контур
    segmented = max(contours0, key=cv.contourArea)
    # упрощаем точки контура до прямоугольника
    epsilon = 0.1*cv.arcLength(segmented,True)
    approx = cv.approxPolyDP(segmented,epsilon,True)     
    # найденные точки контура платы на фото
    src =np.float32(approx)
    # размер изображения на выходе
    dst, width, height = getDest(size['width'],size['height'])
    
    if DrawInfo == True:
        # рисуем прямоугольник
        cv.drawContours(imgin,[approx],0,(100,255,0),2) 
        for aa in approx:
            print(aa)
            xy=(aa[0][0],aa[0][1])
            cv.circle(imgin, xy, 5, color_yellow, 2) # рисуем маленький кружок в центре прямоугольника
            cv.putText(imgin, "%d,%d" % (xy), xy,cv.FONT_HERSHEY_SIMPLEX, 1, color_blue, 2)

    img0, m = unwarp(imgin, src, dst, False)
    print(m)

    crop_img = img0[0:width, 0:height]

    # H,W = crop_img.shape[:2]
    # if W < H:
    #    crop_img= rotateImage(crop_img, 90)

    return crop_img
        
if __name__ == '__main__':
    imgDir = 'pcbs/'
    fileName='pcb1.jpg'
    fn = imgDir+fileName
    img = cv.imread(fn)

    print('Original Dimensions : ',img.shape)
    
    pcbSize = dict(width=500, height=800)
    res = findContours(img,pcbSize)
    update(res)


    # cv.createTrackbar( "rotate", "contours", 0, 4, rotate )
    cv.imwrite('recon_'+fileName, res)
    cv.waitKey()
    cv.destroyAllWindows()