from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
%matplotlib inline
import numpy as np
import cv2 as cv

images = []
threshold, chosen_box, stats, thresh, img = get_best_threshold('full-dataset/dataset7(rgb134).jpg', start=40)
print "threshold used: "+str(threshold)
# chosen_box = cv2.convertScaleAbs(chosen_box)
im2,contours,hierarchy = cv.findContours(thresh, 1, 2)
# plt.imshow(chosen_box, cmap="gray")
# plt.show()
areas = []
for cont in contours:
    areas.append(cv.contourArea(cont))
areas = np.array(areas)
print "max area: "+ str(2700)
cont = contours[np.argmax(areas[areas < 2700])]
rect = cv.minAreaRect(cont)
box = cv.boxPoints(rect)
box = np.int0(box)
images.append(thresh)
a = cv.drawContours(thresh,[box],0,(255, 255,0),2)
plt.imshow(thresh)
plt.show()
print "angle: " +str(rect[2])
# print "length x width: " +str(rect[1])
print "area: "+str(rect[1][0]*rect[1][1])
print "bbox ratio: "+str(max(rect[1][0]/rect[1][1],rect[1][1]/rect[1][0]))

def get_best_threshold(img_filename, start=30, max_steps =100, alpha=1):
    min_ratio = float("inf")
    index = -1
    final_chosen_box = None
    final_stats = []
    final_img = None
    final_thresh = None
    thresholds = np.arange(start,start+max_steps,alpha)
    for i in thresholds:
        stats, chosen_box, thresh, img = get_CCA_from_image(img_filename, i, show_image=False)
        ratio = get_ratio_of_image(chosen_box)
        if ratio < min_ratio:
            min_ratio = ratio
            index = i
            final_chosen_box = chosen_box
            final_stats = stats
            final_thresh = thresh
            final_img = img
    return index, final_chosen_box, final_stats, final_thresh, final_img
def get_ratio_of_image(img):
    #ratio of white to black
    a=cv2.countNonZero(img)
    h,w = img.shape[:2]
#     print type(float(a))
#     print (float(h)*float(w))
    return float(float(a)/(float(h)*float(w)))
def get_CCA_from_image(image, threshold = 115, show_image=False):
    img = cv2.imread(image,0)
    h,w = img.shape[:2]

    # plt.show()
    if show_image:
        fig,ax = plt.subplots(1)
    # ax.imshow(img,cmap='gray')

    L = measure.label(img)
    # print "Number of components:", np.max(L)
    # print L

    ret, thresh = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY_INV)
    
    output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)

    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

#     print num_labels
#     print labels, len(labels)
#     print stats
#     print centroids

    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # put a red dot, size 40, at 2 locations:
    # plt.scatter(x=[centroids[0][0], centroids[1][0]], y=[centroids[0][1], centroids[1][1]], c='r', s=40)
    # ax2 = plt.add_subplot(111, aspect='equal')
    max_area_index=-1
    for i in range(len(stats)):
        if stats[i][0]==0 and stats[i][1]==0:
            continue
        if show_image:
            ax.add_patch(
                patches.Rectangle(
                    (stats[i][0], stats[i][1]),
                    stats[i][2],
                    stats[i][3],
                    fill=False,      # remove background
                    color='red'
                )
            )
        if max_area_index==-1:
            max_area_index=i
        elif stats[i][4] > stats[max_area_index][4]:
            max_area_index=i
            
    # fig2.savefig('rect2.png', dpi=90, bbox_inches='tight')
    if show_image:
        ax.imshow(thresh, cmap='gray')
        plt.show()
    #crop
    chosen_box = thresh[stats[max_area_index][1]:stats[max_area_index][1]+stats[max_area_index][3],stats[max_area_index][0]:stats[max_area_index][0]+stats[max_area_index][2]]
    return stats[max_area_index], chosen_box, thresh, img
