"""Utility functions for SVM regression models."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

def scale_data(data):
    """
    Scales the input data using StandardScaler.
    Parameters:
        data (np.ndarray): The input data to be scaled.
    Returns:
        data_scaled (np.ndarray): The scaled data.
        scaler (StandardScaler): The fitted scaler object.
    """
    # Fit the scaler and transform the data
    scaler = StandardScaler().fit(data)
    # Transform the data
    data_scaled = scaler.transform(data)
    return data_scaled, scaler


def test_model_fit(X_scaled, y_scaled, C_, gamma_, epsilon):
    """
    """
    print(f"C: {C_}; gamma: {gamma_:.2f}; epsilon: {epsilon}")
    regr = SVR(
        kernel='rbf',C=C_, gamma=gamma_,epsilon=epsilon
    )
    SVM_fit_model = regr.fit(X_scaled, y_scaled)
    
    y_pred = SVM_fit_model.predict(X_scaled)
    
    print(f'Training fit R2-score: {r2_score(y_scaled, y_pred):.2f}')
    
    return SVM_fit_model, y_pred



def get_ALE(X, model_fit, feat_index=0, num_intervals = 10):
    """
    """
    # initialize array to save ALE values
    ale_values = np.zeros(num_intervals)
    # feature vector
    X_feat = X[:,feat_index]
    # Create intervals for the chosen feature
    interval_edges = np.linspace(
        X_feat.min(), X_feat.max(), num_intervals + 1
    )
    # loop over the range of feature values defined via intervals 
    for i in range(num_intervals):
        # Find data points within the current interval
        mask = (X_feat >= interval_edges[i]) & (X_feat < interval_edges[i+1])
    
        if np.any(mask):
            X_interval_lower = X[mask,:].copy()
            X_interval_lower[:,feat_index] = X[mask,feat_index].min()
            predictions_lower = model_fit.predict(X_interval_lower)
            
            X_interval_upper = X[mask,:].copy()
            X_interval_upper[:,feat_index] = X[mask,feat_index].max()
            predictions_upper = model_fit.predict(X_interval_upper)
            
            # Average local effect for the interval
            ale_values[i] = np.mean(predictions_upper - predictions_lower)
    
    plt.figure(figsize=(4, 3))
    plt.plot(
        interval_edges[:-1],
        ale_values
    )
    plt.xlabel('Input feature value')
    plt.ylabel('ALE (accumulated local effect)')
    plt.show()