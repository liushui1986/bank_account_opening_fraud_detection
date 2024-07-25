import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('baf_train.csv')
X_train = df.iloc[:, :-1] 
y_train = df.iloc[:, -1]

# Separate the majority and minority classes
X_train_majority = X_train[y_train == 0].copy()
X_train_minority = X_train[y_train == 1].copy()

# Perform clustering on the majority class
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
X_train_majority['cluster'] = kmeans.fit_predict(X_train_majority)

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

# Undersample the majority class
X_majority_undersampled, y_majority_undersampled = undersample_clusters(X_train_majority, target_ratio=0.1)

# Combine the undersampled majority class with the minority class
X_train_combined = pd.concat([X_majority_undersampled, X_train_minority])
y_train_combined = pd.concat([y_majority_undersampled, y_train[y_train == 1]])

# Apply SMOTE to the combined data
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_combined, y_train_combined)

# Define and fit the LightGBM model
lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_train_resampled, y_train_resampled)

# Save the model and transformer
with open('baf_pipeline_deply.pkl', 'wb') as f:
    pickle.dump((tf, kmeans, lgbm, 0.6459), f)