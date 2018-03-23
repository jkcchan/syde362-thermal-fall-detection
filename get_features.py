from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
%matplotlib inline
import numpy as np
import cv2 as cv

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
            
    if show_image:
        ax.imshow(thresh, cmap='gray')
        plt.show()
    #crop
    chosen_box = thresh[stats[max_area_index][1]:stats[max_area_index][1]+stats[max_area_index][3],stats[max_area_index][0]:stats[max_area_index][0]+stats[max_area_index][2]]
    return stats[max_area_index], chosen_box, thresh, img


def get_rotated_box(image_filename, output = False, max_area = 2700):
    images = []
    threshold, chosen_box, stats, thresh, img = get_best_threshold(image_filename, start=40)
    im2,contours,hierarchy = cv.findContours(thresh, 1, 2)
    areas = []
    for cont in contours:
        areas.append(cv.contourArea(cont))
    areas = np.array(areas)
    cont = contours[np.argmax(areas[areas < max_area])]
    rect = cv.minAreaRect(cont)
    if output:
        box = cv.boxPoints(rect)
        box = np.int0(box)
        images.append(thresh)
        a = cv.drawContours(thresh,[box],0,(255, 255,0),2)
        plt.imshow(thresh)
        plt.show()
    return rect[1][0]*rect[1][1], max(rect[1][0]/rect[1][1],rect[1][1]/rect[1][0])