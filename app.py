import gradio as gr
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

from train_model.train import run_one, run_two

with open("C:/Users/hp/Health-Insurance-Cost-Prediction/models/random_forest_model.pickle", "rb") as f:
    random_model = pickle.load(f)

with open("C:/Users/hp/Health-Insurance-Cost-Prediction/models/gradient_boost_regressor.pickle", "rb") as f:
    gradient_model = pickle.load(f)

# Function to make predictions using the classifier model
models = [random_model, gradient_model]
def predict_insurance_cost(age,sex,weight,bmi,hereditary_diseases,no_of_dependents,smoker,bloodpressure,diabetes,regular_ex, model):
    # Preprocess input features if needed
    
    feature_names = ["age","sex","weight","bmi","hereditary_diseases","no_of_dependents","smoker","bloodpressure","diabetes","regular_ex"]
    input_data = {
    'age': age,
    'sex': sex,
    'weight': weight,
    'bmi': bmi,
    'hereditary_diseases': hereditary_diseases,
    'no_of_dependents': no_of_dependents,
    'smoker': smoker,
    'bloodpressure': bloodpressure,
    'diabetes': diabetes,
    'regular_ex': regular_ex
}
    # Convert input data to a DataFrame with a single row
    input_data = pd.DataFrame([input_data])
    input_data = input_data[feature_names]
    
    for col in feature_names:
        input_data[col] = pd.to_numeric(input_data[col])
        
    input_data = input_data.fillna(input_data.median())
    input_data=input_data.drop_duplicates()
    
    # print(input_data)
    
    prediction = models[model].predict(input_data)
    
    return prediction[0]
  
# Create Gradio interface
def app_interface():
    with gr.Blocks() as interface:
        with gr.Row("Health Insurance Cost Prediction"):
            
            with gr.Column("Model Training "):
                gr.HTML("<h2>Train your own model!</h2>")
                with gr.Row("Random Forest Model"):
                    with gr.Row(""):
                        gr.HTML("<h2>Random Forest Model</h2>")
                        
                    random_input = [
                        gr.Slider(minimum=10, maximum=500, step = 5, label="Number of Estimators"), 
                        gr.Slider(minimum=1, maximum=500, step = 1, label="Max Depth"), 
                        gr.Slider(minimum=0, maximum=1, step = 0.1, label="Test Size"), 
                    ]
                    random_output = [
                        gr.Textbox(label="MEA Score"),
                        gr.Textbox(label="R2 Score"), 
                        gr.Image(label="ROC"),
                        gr.Image(label = "Loss")
                        
                    ]
                    
                    random_train_button = gr.Button(value="Train Random Forest Model")
                    
                with gr.Row("Gradient Boost Regressor"):
                    gr.HTML("<h2>Gradient Boost Regressor Model</h2>")
                    gradient_input = [
                        gr.Slider(minimum=10, maximum=500, step = 5, label="Number of Estimators"),
                        gr.Slider(minimum=0.00000000001, maximum=1, label="Learning Rate"),
                        gr.Slider(minimum=1, maximum=500, step = 1, label="Max Depth"), 
                        gr.Slider(minimum=0, maximum=1, step = 0.1, label="Test Size"), 
                    ]
                    gradient_output = [
                        gr.Textbox(label="MEA Score"),
                        gr.Textbox(label="R2 Score"),
                        gr.Image(label="ROC"),
                        gr.Image(label = "Loss")
                    ]
                    
                    gradient_train_button = gr.Button(value="Train Gradient Boost Regressor Model")
                    
                
            with gr.Column("Please fill the form to predict insurance cost!"):
                gr.HTML("<h2>Please fill the form to predict insurance cost!</h2>")
                
                inp = [
                    gr.Slider(label="Age", minimum=1, maximum=120),
                    gr.Radio(label="Sex", choices=[("Male", 1), ("Female", 0)]),
                    gr.Slider(label="Weight (kg)", minimum=10, maximum=300, step=1),
                                
                    gr.Slider(label="BMI", minimum=10, maximum=30, step=1),
                    gr.Radio(label="Hereditary Disease", choices=[
                        ('NoDisease', 0),( 'Epilepsy',1), ('EyeDisease',2), ('Alzheimer',3), ('Arthritis',4),
                        ('HeartDisease',5), ('Diabetes',6), ('Cancer',7), ('High BP',8), ('Obesity',9)
                    ]),
                    gr.Slider(label="Number of Dependents", minimum=0, maximum=20, step=1),
                    
                    gr.Radio(label="Smoking", choices=[("Non Smoker",0), ("Smoker", 1)]),
                    gr.Slider(label="Blood Pressure", minimum=50, maximum=300, step=5),
                    gr.Radio(label="Diabetes", choices=[("Not Present",0), ("Present", 1)]),
                    gr.Radio(label="Physical Activity", choices=[("Non Active", 0), ("Active", 1)]),
                    
                    gr.Radio(label = "Model", choices=[("Random Forest", 0), ("Gradient Boost Regressor", 1)])
                ]
                
                output = [gr.Textbox(label="Prediction")]
                predict_button = gr.Button(value="Health Insurance Cost Prediction")
                
        random_train_button.click(run_one, inputs=random_input, outputs=random_output)
        gradient_train_button.click(run_two, inputs=gradient_input, outputs=gradient_output)
        predict_button.click(predict_insurance_cost, inputs=inp, outputs=output)

    interface.launch()

if __name__ == "__main__":
    app_interface()