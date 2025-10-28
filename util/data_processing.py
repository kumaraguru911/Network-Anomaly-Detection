import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy
import seaborn as sns

import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


class processData():
    def __init__(self, file_path):
        self.raw_data = pd.read_csv(file_path)
        # Strip whitespace from column names to prevent KeyErrors
        # e.g. ' Label' -> 'Label'
        self.raw_data.columns = self.raw_data.columns.str.strip()
        pass

    def save_df(self, predictions,num_samples, file_path):
        self.raw_data["Label"] = predictions[:num_samples] # Changed 'label' to 'Label'
        self.raw_data.to_csv(file_path)

    def visualizeData(self):
        """
            Visualize the data,
        """
        pass
    

    def prepareTrainingData(self):

        # Based on Assumption I set all the data labeled as normal --> 0,
        # everything else as anomaly --> 1
        
        # Print diagnostic information about labels
        print("\nDiagnostic Information:")
        print(f"Total samples in dataset: {len(self.raw_data)}")
        unique_labels = self.raw_data["Label"].unique()
        print(f"Unique labels in dataset: {unique_labels}")
        
        # Count occurrences of each label
        label_counts = self.raw_data["Label"].value_counts()
        print("Label distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
        
        # Define normal labels in lowercase for case-insensitive matching
        normal_label_patterns = ['normal', 'benign']
        
        # Check if any normal labels exist in the data
        # This check is now case-insensitive
        normal_exists = any(str(label).lower() in normal_label_patterns for label in unique_labels)
        
        if not normal_exists:
            print("\nWARNING: No 'normal' or 'BENIGN' labels found in the dataset.")
            print("Treating the most common label as normal traffic.")
            # Find most common label and treat it as normal
            most_common_label = label_counts.idxmax()
            print(f"Using '{most_common_label}' as normal traffic (label=0).")
            # Identify all labels that are not the most common as anomalies
            anomaly_labels = [l for l in unique_labels if l != most_common_label]
            normal_labels_to_replace = [most_common_label]
        else:
            # Identify normal and anomaly labels from the dataset based on patterns
            normal_labels_to_replace = [l for l in unique_labels if str(l).lower() in normal_label_patterns]
            anomaly_labels = [l for l in unique_labels if str(l).lower() not in normal_label_patterns]
        
        # Create a mapping for replacement to address the FutureWarning
        replace_map = {label: 0 for label in normal_labels_to_replace}
        replace_map.update({label: 1 for label in anomaly_labels})
        
        # Apply the mapping
        # Explicitly convert to numeric to avoid FutureWarning
        self.raw_data['Label'] = pd.to_numeric(self.raw_data['Label'].replace(replace_map))
        data = self.raw_data
        
        # Data Cleaning: Handle non-numeric and infinite values
        # The CIC-IDS-2017 dataset has 'Infinity' and NaN values.
        # 1. Replace infinite values with NaN
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 2. Drop the label column before cleaning features
        y = data['Label']
        X = data.drop(['Label'], axis=1)
        
        # 3. Fill NaN with 0 and ensure all feature columns are numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        data = pd.concat([X, y], axis=1)


        #Let's split the data to train and validation
        Train, Val = sklearn.model_selection.train_test_split(data, test_size=0.2, random_state=1, shuffle=True)
        
        # Let's now split the features and the target ground truth
        features = [feature for feature in Train.columns.tolist() if feature not in ["Label"]] # Use "Label" here too
        target = "Label" # Use "Label" here too

        # Define a random state 
        state = np.random.RandomState(42)
        X_train = Train[features]
        Y_train = Train[target]

        X_val = Val[features]
        Y_val = Val[target]

        X_outliers = state.uniform(low=0, high=1, size=(X_train.shape[0], X_train.shape[1]))

        # Scale the data
        scaler = sklearn.preprocessing.MinMaxScaler() # Use MinMaxScaler for consistency with autoencoder.py
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        print("Data processed ...")
        return (X_train_scaled, Y_train.values, X_val_scaled, Y_val.values, scaler)

    def prepareTestData(self, scaler):
            data = self.raw_data
            
            # Drop the 'Unnamed: 0' column if it exists from reading test_data.csv
            if 'Unnamed: 0' in data.columns:
                data = data.drop('Unnamed: 0', axis=1)

            # Separate features and labels (if 'Label' exists)
            if 'Label' in data.columns:
                X = data.drop('Label', axis=1)
            else:
                X = data
            
            # Apply the same cleaning steps as in prepareTrainingData
            # 1. Replace infinite values with NaN
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # 2. Fill NaN with 0 and ensure all feature columns are numeric
            # This handles 'Infinity', NaN, and any other non-numeric strings
            X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

            # Scale the data using the scaler from training
            data_scaled = scaler.transform(X)
            return data_scaled
    