import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

def load_images(folder):
    data = []
    labels = []
    for label in ['yes', 'no']:
        path = os.path.join(folder, label)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Skipped: {file_path} (not a valid image)")
                continue  # skip bad files

            img = cv2.resize(img, (100, 100)).flatten()
            data.append(img)
            labels.append(1 if label == 'yes' else 0)
    
    return np.array(data), np.array(labels)


X, y = load_images('dataset')  # your dataset/yes and dataset/no folders
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')
