import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('baf_train_07262024_2_sm.csv')
X = df.iloc[:, :-1] 
y = df.iloc[:, -1]

# Separate the majority and minority classes
X_majority = X[y == 0].copy()
X_minority = X[y == 1].copy()

# Perform clustering on the majority class
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
X_majority['cluster'] = kmeans.fit_predict(X_majority)

# Function to undersample within each cluster
def undersample_clusters(X_majority, target_ratio=0.1):
    undersampled_X = pd.DataFrame()
    undersampled_y = pd.Series(dtype='int')
    
    for cluster in X_majority['cluster'].unique():
        cluster_data = X_majority[X_majority['cluster'] == cluster]
        cluster_size = int(len(cluster_data) * target_ratio)
        undersampled_cluster = cluster_data.sample(cluster_size, random_state=42)
        
        undersampled_X = pd.concat([undersampled_X, undersampled_cluster])
        undersampled_y = pd.concat([undersampled_y, pd.Series([0] * cluster_size)])
    
    return undersampled_X.drop('cluster', axis=1), undersampled_y

# Use the best ratio for undersampling
X_majority_undersampled, y_majority_undersampled = undersample_clusters(X_majority, target_ratio=0.1)

# Combine the undersampled majority class with the minority class
X_combined = pd.concat([X_majority_undersampled, X_minority])
y_combined = pd.concat([y_majority_undersampled, y[y == 1]])

# Apply SMOTE to the combined data
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_combined, y_combined)

# Define and fit the LightGBM model
lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_resampled, y_resampled)

# Save the model and transformer
with open('baf_model_sm_0726_2.pkl', 'wb') as f:
    pickle.dump((kmeans, lgbm, 0.6459), f)