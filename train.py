import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

def get_CCA_from_image(image, threshold = 115, show_image=False):
    img = cv2.imread(image,0)
    h,w = img.shape[:2]
    if show_image:
        fig,ax = plt.subplots(1)

    L = measure.label(img)

    ret, thresh = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY_INV)
    
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

def get_features(filename, output=False):
    index, final_chosen_box, final_stats, final_thresh, final_img = get_best_threshold(filename)
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
    
def scale_features(features):
    if not scl:
        return
    if len(features) != (scl.min_).shape[0]:
        return
    return scl.transform([features])

def classify_with_scaled_features(scaled_features, probabilities = False):
    if not clf:
        return
    if probabilities:
        return clf.predict(scaled_features), clf.predict_proba(scaled_features)
    return clf.predict(scaled_features)

def train():
    import pandas as pd
    from sklearn import datasets, linear_model
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
    from sklearn.preprocessing import MinMaxScaler
    from sklearn import svm
    from sklearn import gaussian_process
    from sklearn import discriminant_analysis
    from sklearn import neighbors

    classifiers = [linear_model.LogisticRegression(),
                   svm.SVC(), 
                   svm.SVC(gamma=2, C=1), 
                   gaussian_process.GaussianProcessClassifier(), 
                   discriminant_analysis.QuadraticDiscriminantAnalysis(),
                   neighbors.KNeighborsClassifier()]
    performance = {}

    def classify(clf, X_train, X_test, y_train, y_test):
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        return clf, accuracy_score(y_test, prediction), mean_squared_error(y_test, prediction),r2_score(y_test, prediction)

    for i in classifiers:
        performance[str(type(i))] = {'scaler':[],'clf':[], 'accuracy_score':[], 'mse' :[], 'r2':[]}
    for j in range(0,100):
        train = pd.read_csv('regression_3.csv')
        test = pd.read_csv('regression_3_test.csv')
        data = train.append(test)
        data = data.reset_index().drop(['index'], axis=1)
        y = data.falling
        X = data.drop(['falling', 'position'], axis=1)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit(X)
        X_scaled = scaler.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
        for i in classifiers:
            clf, accuracy_score_value, mse_value, r2_value = classify(i,X_train, X_test, y_train, y_test)
            performance[str(type(i))]['clf'].append(clf)
            performance[str(type(i))]['accuracy_score'].append(accuracy_score_value)
            performance[str(type(i))]['mse'].append(mse_value)
            performance[str(type(i))]['r2'].append(r2_value)
            performance[str(type(i))]['scaler'].append(scaler)
    for i in classifiers:
        performance[str(type(i))]['avg_score'] = np.average(performance[str(type(i))]['accuracy_score'])
        performance[str(type(i))]['avg_mse'] = np.average(performance[str(type(i))]['mse'])
        performance[str(type(i))]['avg_r2'] = np.average(performance[str(type(i))]['r2'])
    
    best_class = None
    best_score = -1
    for i in performance:
        if performance[i]['avg_score'] > best_score:
            best_score = performance[i]['avg_score']
            best_class = i
    classifier = performance[best_class]\
    ['clf'][np.argmax(performance[best_class]['accuracy_score'])]
    scaler = performance[best_class]\
    ['scaler'][np.argmax(performance[best_class]['accuracy_score'])]
    return classifier, scaler

def export_to_pickles(clf, scl):
    from sklearn.externals import joblib
    joblib.dump(classifier, 'classifier.pkl') 
    joblib.dump(scaler, 'scaler.pkl') 