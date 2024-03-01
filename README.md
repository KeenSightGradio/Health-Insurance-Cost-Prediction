# Health Insurance Cost Prediction

This project aims to predict health insurance costs based on various input features such as age, sex, weight, BMI, hereditary diseases, number of dependents, smoking status, blood pressure, diabetes, and physical activity. The prediction is made using two trained models: Random Forest Model and Gradient Boost Regressor Model.

## Installation

To run this project, you'll need to install the following dependencies:

- gradio
- numpy
- pandas
- scikit-learn

You can install the dependencies by running the following command:

## Usage

To use the Health Insurance Cost Prediction application, follow these steps:

1. Run the `app_interface()` function in the `main.py` file.
2. Fill out the form with the required information, such as age, sex, weight, BMI, etc.
3. Choose the desired model (Random Forest or Gradient Boost Regressor) for prediction.
4. Click on the "Health Insurance Cost Prediction" button to get the predicted insurance cost.

## Model Training

If you want to train your own models, you can use the provided functionalities:

- Random Forest Model Training:
    - Adjust the number of estimators using the slider.
    - Click on the "Train Random Forest Model" button to train the model.
    - The MEA score and R2 score will be displayed.

- Gradient Boost Regressor Model Training:
    - Adjust the number of estimators and learning rate using the sliders.
    - Click on the "Train Gradient Boost Regressor Model" button to train the model.
    - The MEA score and R2 score will be displayed.

## Note

Make sure to have the trained model files (`random_forest_model.pickle` and `gradient_boost_regressor.pickle`) in the correct directory before running the application.

