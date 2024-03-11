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

def load_dataset(dataset_path, test_size):
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test 

def train_model_one(train_data, train_label,n_estimators, max_depth):
    # Initialize and train a Random Forest Regressor model
    model = RandomForestRegressor(n_estimators = n_estimators,max_depth = max_depth )
    model.fit(train_data, train_label)

    with open("C:/Users/hp/Health-Insurance-Cost-Prediction/models/random_forest_model.pickle", "wb") as f:
        pickle.dump(model, f)


def test_model_one(test_data, test_label):
    with open("C:/Users/hp/Health-Insurance-Cost-Prediction/models/random_forest_model.pickle", "rb") as f:
        loaded_model = pickle.load(f)
        
    y_pred = loaded_model.predict(test_data)
    
    # Calculate and store the Mean Absolute Error (MAE) for the Random Forest Regressor (RF) model
    mae_model = metrics.mean_absolute_error(test_label, y_pred)
    r2_score = metrics.r2_score(test_label, y_pred)
    
    return y_pred, mae_model, r2_score

def plot(y_pred, test_label):

    data = pd.DataFrame({'Actual': test_label, 'Predicted': y_pred})
    
    plt.plot(data['Actual'].iloc[0:11], label='Actual')
    plt.plot(data['Predicted'].iloc[0:11], label="Predicted")
    plt.xticks(data['Actual'].iloc[0:11], rotation=45)
    plt.ylabel('Cost')
    
    plt.legend()
    plt.tight_layout()
    plt.title('Random Forest Predictions')
    
    # Convert the plot to an image
    fig = plt.gcf()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return image

def visualize_feature_importances(feature_names):
    with open("C:/Users/hp/Health-Insurance-Cost-Prediction/models/random_forest_model.pickle", "rb") as f:
        model = pickle.load(f)
        
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]
    sorted_feature_names = feature_names[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), sorted_importances, tick_label=sorted_feature_names)
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    # plt.show()
    
    # Convert the plot to an image
    fig = plt.gcf()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return image

def run_one(n_estimators, max_depth, test_size):
    
    file_path = "C:/Users/hp/Health-Insurance-Cost-Prediction/dataset/healthinsurance.csv"
    train_data,  test_data, train_label, test_label = load_dataset(file_path, test_size)
    
    train_model_one(train_data, train_label, n_estimators, max_depth)
    y_pred, MAE_score_one, R2_Score_one = test_model_one(test_data, test_label)
    
    feat_impo = visualize_feature_importances(train_data.columns)
    plot_image = plot(y_pred, test_label)
    
    visuals = [feat_impo, plot_image]
  
    return MAE_score_one, R2_Score_one, visuals

  
if __name__ == "__main__":
    run_one(n_estimators=10, max_depth=5, test_size=0.2)
    
    
    
    