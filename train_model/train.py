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
        
def train_model_two(train_data, train_label, n_estimators, learning_rate, max_depth):
    # Initialize and train a Random Forest Regressor model
    model = GradientBoostingRegressor(n_estimators = n_estimators, learning_rate = learning_rate, max_depth = max_depth)
    model.fit(train_data, train_label)

    with open("C:/Users/hp/Health-Insurance-Cost-Prediction/models/gradient_boost_regressor.pickle", "wb") as f:
        pickle.dump(model, f)


def test_model_one(test_data, test_label):
    with open("C:/Users/hp/Health-Insurance-Cost-Prediction/models/random_forest_model.pickle", "rb") as f:
        loaded_model = pickle.load(f)
        
    y_pred = loaded_model.predict(test_data)
    
    # Calculate and store the Mean Absolute Error (MAE) for the Random Forest Regressor (RF) model
    mae_model = metrics.mean_absolute_error(test_label, y_pred)
    r2_score = metrics.r2_score(test_label, y_pred)
    
    return mae_model, r2_score
 
def test_model_two(test_data, test_label):
    
    with open("C:/Users/hp/Health-Insurance-Cost-Prediction/models/gradient_boost_regressor.pickle", "rb") as f:
        loaded_model = pickle.load(f)
        
    y_pred = loaded_model.predict(test_data)
    y_pred_proba = loaded_model.predict_proba(test_data)[:, 1]
    mae_model = metrics.mean_absolute_error(test_label, y_pred)
    r2_score = metrics.r2_score(test_label, y_pred)
    
    return y_pred_proba, mae_model, r2_score


def run_one(n_estimators, max_depth, test_size):
    
    file_path = "C:/Users/hp/Health-Insurance-Cost-Prediction/dataset/healthinsurance.csv"
    train_data,  test_data, train_label, test_label = load_dataset(file_path, test_size)
    
    train_model_one(train_data, train_label, n_estimators)
    _, MAE_score_one, R2_Score_one = test_model_one(test_data, test_label)
    
    roc_curve = plot_roc_curve_one(test_data, test_label)
    learning_curve = plot_learning_curve_one(train_data, train_label)
    
    return MAE_score_one, R2_Score_one, roc_curve, learning_curve

def run_two(n_estimators, learning_rate,max_depth, test_size):
    
    file_path = "C:/Users/hp/Health-Insurance-Cost-Prediction/dataset/healthinsurance.csv"
    train_data,  test_data, train_label, test_label = load_dataset(file_path, test_size)
    
    train_model_two(train_data, train_label, n_estimators, learning_rate, max_depth)
    _, MAE_score_two, R2_Score_two = test_model_two(test_data, test_label)
    
    roc_curve = plot_roc_curve_two(test_data, test_label)
    learning_curve = plot_learning_curve_two(train_data, train_label)
    
    return MAE_score_two, R2_Score_two, roc_curve, learning_curve

def plot_learning_curve_one(train_data, train_label):
    # Load the trained model
    with open("C:/Users/hp/Credit-Default-Risk-Prediction/models/random_forest_model.pickle", "rb") as f:
        loaded_model = pickle.load(f)

    train_errors = []
    test_errors = []
    train_sizes = [0.125, 0.25,0.5]

    for train_size in train_sizes:
        # Split the training data into smaller training set and validation set
        X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, train_size=train_size, random_state=42)

        # Train the model on the smaller training set
        loaded_model.fit(X_train, y_train)

        # Predict on the training set and calculate the training error
        y_train_pred = loaded_model.predict(X_train)
        train_errors.append(metrics.mean_squared_error(y_train, y_train_pred))

        # Predict on the validation set and calculate the testing error
        y_val_pred = loaded_model.predict(X_val)
        test_errors.append(metrics.mean_squared_error(y_val, y_val_pred))
        
     # Plot the learning curve
    plt.figure()
    plt.plot(train_sizes, train_errors, 'o-', color="r", label="Training error")
    plt.plot(train_sizes, test_errors, 'o-', color="g", label="Validation error")
    plt.xlabel("Training examples")
    plt.ylabel("Error")
    plt.title("Learning Curve")
    plt.legend(loc="best")

    # Convert the plot to an image
    fig = plt.gcf()
    fig.canvas.draw()
    learning_curve_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    learning_curve_image = learning_curve_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return learning_curve_image

def plot_learning_curve_two(train_data, train_label, test_data, test_label):
    # Load the trained model
    with open("C:/Users/hp/Credit-Default-Risk-Prediction/models/gradient_boost_regressor.pickle", "rb") as f:
        loaded_model = pickle.load(f)

    train_errors = []
    test_errors = []
    train_sizes = [0.125, 0.25,0.5]

    for train_size in train_sizes:
        # Split the training data into smaller training set and validation set
        X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, train_size=train_size, random_state=42)

        # Train the model on the smaller training set
        loaded_model.fit(X_train, y_train)

        # Predict on the training set and calculate the training error
        y_train_pred = loaded_model.predict(X_train)
        train_errors.append(metrics.mean_squared_error(y_train, y_train_pred))

        # Predict on the validation set and calculate the testing error
        y_val_pred = loaded_model.predict(X_val)
        test_errors.append(metrics.mean_squared_error(y_val, y_val_pred))
        
     # Plot the learning curve
    plt.figure()
    plt.plot(train_sizes, train_errors, 'o-', color="r", label="Training error")
    plt.plot(train_sizes, test_errors, 'o-', color="g", label="Validation error")
    plt.xlabel("Training examples")
    plt.ylabel("Error")
    plt.title("Learning Curve")
    plt.legend(loc="best")

    # Convert the plot to an image
    fig = plt.gcf()
    fig.canvas.draw()
    learning_curve_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    learning_curve_image = learning_curve_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return learning_curve_image
def plot_roc_curve_one(test_data, test_label):
    # Get predicted probabilities
    y_pred_proba, _, _, _, _ = test_model_one(test_data, test_label)
    
    # Calculate the ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(test_label, y_pred_proba)
    roc_auc = metrics.auc(fpr, tpr)
    
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    # Convert the plot to an image
    fig = plt.gcf()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return image

def plot_roc_curve_two(test_data, test_label):
    # Get predicted probabilities
    y_pred_proba, _, _, _, _ = test_model_two(test_data, test_label)
    
    # Calculate the ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(test_label, y_pred_proba)
    roc_auc = metrics.auc(fpr, tpr)
    
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    # Convert the plot to an image
    fig = plt.gcf()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return image
  
if __name__ == "__main__":
    run_one(n_estimators=10, max_depth=5, test_size=0.2)
    run_two(n_estimators=10, learning_rate=0.001, max_depth = 5, test_size=0.2)
    
    
    