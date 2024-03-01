import numpy as np
import pandas as pd 
import os
import itertools
# import seaborn as sns
import pickle
# sns.set(color_codes=True)

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import metrics

def load_dataset(dataset_path):
    data = pd.read_csv(dataset_path) 
    
    # Initialize the SimpleImputer with the strategy 'mean'
    imputer = SimpleImputer(strategy='mean')

    # Specify the columns with missing values that we want to impute
    columns_with_missing_values = data.columns[data.isna().any()].tolist()

    # Apply the imputer to fill missing values with the mean
    data[columns_with_missing_values] = imputer.fit_transform(data[columns_with_missing_values])
    
    #values mapped to integers (0 and 1)
    data['sex']=data['sex'].map({'female':0,'male':1})
    data["hereditary_diseases"]=data["hereditary_diseases"].map({'NoDisease':0, 'Epilepsy':1, 'EyeDisease':2, 'Alzheimer':3, 'Arthritis':4,
       'HeartDisease':5, 'Diabetes':6, 'Cancer':7, 'High BP':8, 'Obesity':9})

    # label_encoder = LabelEncoder()
    # data['city'] = label_encoder.fit_transform(data['city'])
    # data['job_title'] = label_encoder.fit_transform(data['job_title'])
    
    # with open("C:/Users/hp/GradioApps/Health-Insurance-Cost-Prediction/models/label_encoder.pickle", "wb") as f:
    #     pickle.dump(label_encoder, f)
    
    data=data.drop(['city'],axis=1)
    data=data.drop(['job_title'],axis=1)
    
    X=data.drop(['claim'],axis=1)
    y=data['claim']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test 

def train_model_one(train_data, train_label,n_estimators):
    # Initialize and train a Random Forest Regressor model
    model = RandomForestRegressor(n_estimators = 10)
    model.fit(train_data, train_label)

    with open("C:/Users/hp/Health-Insurance-Cost-Prediction/models/random_forest_model.pickle", "wb") as f:
        pickle.dump(model, f)
        
def train_model_two(train_data, train_label, n_estimators, learning_rate):
    # Initialize and train a Random Forest Regressor model
    model = GradientBoostingRegressor(n_estimators = 10, learning_rate = 0.001)
    model.fit(train_data, train_label)

    with open("C:/Users/hp/Health-Insurance-Cost-Prediction/models/gradient_boost_regressor.pickle", "wb") as f:
        pickle.dump(model, f)


def test_model_one(test_data, test_label):
    with open("C:/Users/hp/Health-Insurance-Cost-Prediction/models/random_forest_model.pickle", "rb") as f:
        loaded_model = pickle.load(f)
        
    y_pred = loaded_model.predict(test_data)
    # print(y_pred)
    
    # Calculate and store the Mean Absolute Error (MAE) for the Random Forest Regressor (RF) model
    mae_model = metrics.mean_absolute_error(test_label, y_pred)
    r2_score = metrics.r2_score(test_label, y_pred)
    
    return mae_model, r2_score
 
 
def test_model_two(test_data, test_label):
    
    with open("C:/Users/hp/Health-Insurance-Cost-Prediction/models/gradient_boost_regressor.pickle", "rb") as f:
        loaded_model = pickle.load(f)
        
    y_pred = loaded_model.predict(test_data)
    mae_model = metrics.mean_absolute_error(test_label, y_pred)
    r2_score = metrics.r2_score(test_label, y_pred)
    
    return mae_model, r2_score
  
def run_one(n_estimators):
    
    file_path = "C:/Users/hp/Health-Insurance-Cost-Prediction/dataset/healthinsurance.csv"
    train_data,  test_data, train_label, test_label = load_dataset(file_path)
    
    train_model_one(train_data, train_label, n_estimators)
    MAE_score_one, R2_Score_one = test_model_one(test_data, test_label)
    return MAE_score_one, R2_Score_one


def run_two(n_estimators, learning_rate):
    
    file_path = "C:/Users/hp/Health-Insurance-Cost-Prediction/dataset/healthinsurance.csv"
    train_data,  test_data, train_label, test_label = load_dataset(file_path)
    
    train_model_two(train_data, train_label, n_estimators, learning_rate)
    MAE_score_two, R2_Score_two = test_model_two(test_data, test_label)
    return MAE_score_two, R2_Score_two
    
if __name__ == "__main__":
    
    run_one(n_estimators=10)
    run_two(n_estimators=10, learning_rate=0.001)
    
    
    