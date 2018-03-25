import fall_detection_min
from sklearn.externals import joblib
clf = joblib.load('classifier2.pkl')
scl = joblib.load('scaler2.pkl')
features = get_features('full-dataset/dataset1(rgb22).jpg')
scaled_features = scale_features(features)
classify_with_scaled_features(scaled_features, True)
