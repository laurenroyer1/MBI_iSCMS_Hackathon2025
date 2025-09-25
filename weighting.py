# MBI_iSCMS_Hackathon2025
import pandas as pd

# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay

from sklearn.neighbors import NearestNeighbors
from numpy import deg2rad

import matplotlib.pyplot as plt

#### import train/test
traindf = pd.read_csv('training_predictionadded_neighboradded.csv')
validationdf = pd.read_csv('validation_neighboradded.csv')
####define target params here once we want to start weighting

# Features and target
TrainFeatures = traindf[["spec_no", "mz", "intensity", "closest_neighbor_distance"]]
TrainSigNoise = traindf["signal"]  # 0 or 1

ValidationFeatures = validationdf[["spec_no", "mz", "intensity", "closest_neighbor_distance"]]
ValidationSigNoise = validationdf["signal"]  # 0 or 1

# Build logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(TrainFeatures, TrainSigNoise)

# Predictions
SigNoise_pred = model.predict(ValidationFeatures)

# Evaluate
print("Accuracy:", accuracy_score(ValidationSigNoise, SigNoise_pred))
print(classification_report(ValidationSigNoise, SigNoise_pred))

# If you want probabilities instead of labels:
y_proba = model.predict_proba(ValidationFeatures)[:, 1]



# ### visualize prediction across parameters to determine areas to target for weighting
# colors = ['green' if p == v else 'red' for p, v in zip(bigvaldf2['signal'], bigvaldf2['predicted_signal'])]#zip(SigNoise_pred, ValidationSigNoise)]
# green_patch = plt.scatter([], [], c='green', label='Correct Noise Label', s=100)
# red_patch = plt.scatter([], [], c='red', label='Incorrect', s=100)
# plt.legend(handles=[green_patch, red_patch])
# # #### mz vs intensity
# # plt.scatter(bigvaldf2['mz'], bigvaldf2['intensity'], c=colors, s=100, alpha=0.7, edgecolors='black')
# # plt.title('Validation Accuracy for m/z vs Intensity')
# # plt.xlabel('m/z')
# # plt.ylabel('Intensity')

# # #### mz vs spec_no
# # plt.scatter(bigvaldf2['mz'], bigvaldf2['spec_no'], c=colors, s=100, alpha=0.7, edgecolors='black')
# # plt.title('Validation Accuracy for m/z vs spec_no')
# # plt.xlabel('m/z')
# # plt.ylabel('spec_no')

# # #### spec_no vs intensity
# # plt.scatter(bigvaldf2['spec_no'], bigvaldf2['intensity'], c=colors, s=100, alpha=0.7, edgecolors='black')
# # plt.title('Validation Accuracy for spec_no vs Intensity')
# # plt.xlabel('spec_no')
# # plt.ylabel('Intensity')

# # #### euclidean SpecNo+mz vs intensity
# # plt.scatter(bigvaldf2['closest_neighbor_distance'], bigvaldf2['intensity'], c=colors, s=100, alpha=0.7, edgecolors='black')
# # plt.title('Validation Accuracy for Closest Spec+MZ Neighbor vs Intensity')
# # plt.xlabel('euclidean SpecNo+mz')
# # plt.ylabel('Intensity')

# #### euclidean SpecNo+mz vs mz
# plt.scatter(bigvaldf2['closest_neighbor_distance'], bigvaldf2['mz'], c=colors, s=100, alpha=0.7, edgecolors='black')
# plt.title('Validation Accuracy for Closest Spec+MZ Neighbor vs mz')
# plt.xlabel('euclidean SpecNo+mz')
# plt.ylabel('mz')


# plt.grid(True)
# plt.show()