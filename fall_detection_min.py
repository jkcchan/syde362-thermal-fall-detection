import numpy as np
import cv2
from scipy.stats import rayleigh
import scipy
def get_CCA_from_image(image, threshold = 115, show_image=False):
    img = cv2.imread(image,0)
    h,w = img.shape[:2]
    if show_image:
        fig,ax = plt.subplots(1)

    L = measure.label(img)

    ret, thresh = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
    
    output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
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
    chosen_box = thresh[stats[max_area_index][1]:stats[max_area_index][1]+stats[max_area_index][3],stats[max_area_index][0]:stats[max_area_index][0]+stats[max_area_index][2]]
    return stats[max_area_index], chosen_box, thresh, img


def get_best_threshold(img_filename, pdf, start=30, max_steps =100, alpha=1, show_graph = False):
    index = -1
    final_chosen_box = None
    final_stats = []
    final_img = None
    final_thresh = None
    thresholds = np.arange(start,start+max_steps,alpha)
    min_score = -1
    if show_graph:
        areas = []
        ratios = []
        scores=[]
    for i in thresholds:
        stats, chosen_box, thresh, img = get_CCA_from_image(img_filename, i, show_image=False)
        ratio = get_ratio_of_image(chosen_box)
        score= (1-ratio)*pdf[chosen_box.size]
        if show_graph:
            scores.append(score)
            areas.append(chosen_box.size)
            ratios.append(ratio)
        if score > min_score:
            min_score = score
            index = i
            final_chosen_box = chosen_box
            final_stats = stats
            final_thresh = thresh
            final_img = img
    if show_graph:
        plt.title('ratios')
        plt.xlabel('threshold value')
        plt.ylabel('ratio of white pixels to black pixels')
        plt.plot(ratios)
        plt.show()
        plt.title('areas')
        plt.xlabel('threshold value')
        plt.ylabel('area of bounding box')
        plt.plot(areas)
        plt.show()
        plt.title('scores')
        plt.xlabel('threshold value')
        plt.ylabel('score')
        plt.plot(scores)
        return index, final_chosen_box, final_stats, final_thresh, final_img, areas,ratios
    return index, final_chosen_box, final_stats, final_thresh, final_img, score

def get_ratio_of_image(img):
    a=cv2.countNonZero(img)
    h,w = img.shape[:2]
    return float(float(a)/(float(h)*float(w)))

def get_rotated_box(image_filename, output = False, max_area = 2700):
    images = []
    threshold, chosen_box, stats, thresh, img = get_best_threshold(image_filename, start=40)
    im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
    areas = []
    for cont in contours:
        areas.append(cv2.contourArea(cont))
    areas = np.array(areas)
    cont = contours[np.argmax(areas[areas < max_area])]
    rect = cv2.minAreaRect(cont)
    if output:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        images.append(thresh)
        a = cv2.drawContours(thresh,[box],0,(255, 255,0),2)
        plt.imshow(thresh, cmap="gray")
        plt.show()
    return rect[1][0]*rect[1][1], max(rect[1][0]/rect[1][1],rect[1][1]/rect[1][0])

def get_features(filename, pdf, output=False):
    index, final_chosen_box, final_stats, final_thresh, final_img, score = get_best_threshold(filename, pdf)
    features = []
    area = final_stats[2]*final_stats[3]
    features.append(area)
    bb_ratio =max((float(final_stats[2])/float(final_stats[3])),(float(final_stats[3])/float(final_stats[2])))
    features.append(bb_ratio)
    hu_moments = cv2.HuMoments(cv2.moments(final_chosen_box))
    features.extend(hu_moments.T[0].tolist())
    variance = np.var(final_chosen_box)
    features.append(variance) 
    if output:
        print "threshold used: " + str(index)
        plt.imshow(final_chosen_box, cmap="gray")
        plt.show()
        print "area: "+str(area)
        print "bb_ratio: "+ str(bb_ratio)
        print "hu_moments: "+np.array2string(hu_moments.T[0])
        print "variance: "+str(variance)
    return features

def scale_features(features, scl):
    if not scl:
        return
    if len(features) != (scl.min_).shape[0]:
        return
    return scl.transform([features])

def classify_with_scaled_features(scaled_features, clf, probabilities = False):
    if not clf:
        return
    if probabilities:
        return clf.predict(scaled_features), clf.predict_proba(scaled_features)
    return clf.predict(scaled_features)