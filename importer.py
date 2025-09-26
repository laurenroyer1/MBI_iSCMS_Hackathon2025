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
traindf = pd.read_csv('train.csv')
validationdf = pd.read_csv('validation.csv')
finaldf = pd.read_csv('test.csv')
####define target params here once we want to start weighting

# Features and target
TrainFeatures = traindf[["spec_no", "mz", "intensity"]]
TrainSigNoise = traindf["signal"]  # 0 or 1

ValidationFeatures = validationdf[["spec_no", "mz", "intensity"]]
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

finalsn = model.predict(finaldf)
finalresult = pd.Series(finalsn, name='signal')
finalresultint = finalresult.fillna(0)
submitdf = pd.merge(finaldf, finalresultint, left_index=True, right_index=True)
submitdfna = submitdf.fillna(0)
print(submitdfna)
submitdfna.to_csv('MBI_Uribre_Royer_final.csv')
###basic performance checks
# RocCurveDisplay.from_estimator(model, ValidationFeatures, ValidationSigNoise)
# plt.title("ROC Curve")
# plt.show()
# PrecisionRecallDisplay.from_estimator(model, ValidationFeatures, ValidationSigNoise)
# plt.title("Precision-Recall Curve")
# plt.show()

# # You can also visualize the results (optional)
# import matplotlib.pyplot as plt
# plt.scatter(TrainFeatures, TrainSigNoise, label='Actual Test Data')
# plt.plot(ValidationFeatures, SigNoise_pred, color='red', label='Model Predictions')
# plt.xlabel('Feature (X)')
# plt.ylabel('SigNoise Accuracy')
# plt.title('Linear Regression with Noise')
# plt.legend()
# plt.show()

# #### turn results into validation df to start testing metrics
# SigNoiseAdd = pd.Series(SigNoise_pred, name='predicted_signal')
# bigvaldf = pd.concat([validationdf, SigNoiseAdd], axis=1, join='outer')
# # print(bigvaldf)

# ###treating spectra number and m/z like coodinates
# def find_closest_neighbor(df, lat_col, lon_col, metric='euclidean'): 
#     """
#     Finds the closest neighboring point for each point in a DataFrame.

#     Args:
#         df (pd.DataFrame): The input DataFrame containing point coordinates.
#         lat_col (str): The name of the column containing latitude values.
#         lon_col (str): The name of the column containing longitude values.
#         metric (str, optional): The distance metric to use from sklearn.
#                                 'euclidean' for Cartesian coordinates.

#     Returns:
#         pd.DataFrame: The original DataFrame with two new columns:
#                       'closest_neighbor_index' (index of the closest neighbor)
#                       and 'closest_neighbor_distance' (distance to the closest neighbor).
#     """

#     if metric == 'haversine':
#         # Convert degrees to radians for haversine metric
#         data = deg2rad(df[[lat_col, lon_col]])
#     else:
#         data = df[[lat_col, lon_col]]

#     # Initialize NearestNeighbors with n_neighbors=2 to get self and closest neighbor
#     neigh = NearestNeighbors(n_neighbors=2, metric=metric)
#     neigh.fit(data)

#     # Find the k-neighbors and their distances
#     distances, indices = neigh.kneighbors(data, n_neighbors=2, return_distance=True)
#     # The first neighbor is always the point itself (distance 0), so we take the second one
#     # The index of the closest neighbor is at indices[:, 1]
#     # The distance to the closest neighbor is at distances[:, 1]
#     df['closest_neighbor_index'] = indices[:, 1]
#     df['closest_neighbor_distance'] = distances[:, 1]

#     return df

# bigvaldf2= find_closest_neighbor(bigvaldf, 'mz', 'spec_no', metric='euclidean')

# # bigvaldf2['neighborindexdistance'] = abs(bigvaldf2[]-bigvaldf2['']) ###working to add
# # bigvaldf2.to_csv('validation_predictionadded_neighboradded.csv')

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

# #### euclidean SpecNo+mz vs intensity
# plt.scatter(bigvaldf2['closest_neighbor_distance'], bigvaldf2['intensity'], c=colors, s=100, alpha=0.7, edgecolors='black')
# plt.title('Validation Accuracy for Closest Spec+MZ Neighbor vs Intensity')
# plt.xlabel('euclidean SpecNo+mz')
# plt.ylabel('Intensity')

# # #### euclidean SpecNo+mz vs mz
# # plt.scatter(bigvaldf2['closest_neighbor_distance'], bigvaldf2['mz'], c=colors, s=100, alpha=0.7, edgecolors='black')
# # plt.title('Validation Accuracy for Closest Spec+MZ Neighbor vs mz')
# # plt.xlabel('euclidean SpecNo+mz')
# # plt.ylabel('mz')


# plt.grid(True)
# plt.show()