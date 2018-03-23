import numpy as np
import cv2 as cv
images = []
threshold, stats, chosen_box, thresh = get_best_threshold('full-dataset/dataset1(rgb50).jpg')
print "threshold used: "+str(threshold)
im2,contours,hierarchy = cv.findContours(thresh, 1, 2)
areas = []
for cont in contours:
    areas.append(cv.contourArea(cont))
areas = np.array(areas)
print "max area: "+ str(2000)
cont = contours[np.argmax(areas[areas < 2000])]
rect = cv.minAreaRect(cont)
box = cv.boxPoints(rect)
box = np.int0(box)
images.append(img)
a = cv.drawContours(images[-1],[box],0,(0,0,255),2)
plt.imshow(a, cmap="gray")
plt.show()
print "angle: " +str(rect[2])
# print "length x width: " +str(rect[1])
print "area: "+str(rect[1][0]*rect[1][1])
print "bbox ratio: "+str(max(rect[1][0]/rect[1][1],rect[1][1]/rect[1][0]))