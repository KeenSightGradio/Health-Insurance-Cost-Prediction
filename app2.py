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
    # print(input_data)
    
    # Convert input data to a DataFrame with a single row
    input_data = pd.DataFrame([input_data])
    input_data = input_data[feature_names]
    for col in feature_names:
        input_data[col] = pd.to_numeric(input_data[col])
        
    input_data = input_data.fillna(input_data.median())
    input_data=input_data.drop_duplicates()
    
    # print(input_data)
    
    prediction = models[model].predict(input_data)
    
    return str(prediction[0]) + "$ cost"


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
random_forest = gr.Interface(
    fn = run_one,
    inputs = random_input,
    outputs = random_output,  
    submit_btn = "Train Model 1",
    title="Train your own model!",
    description="<img src='https://i.ibb.co/Bw08434/logo-1.png' alt='Logo' style='width:230px;height:100px;border-radius:5px;box-shadow:2px 2px 5px 0px rgba(0,0,0,0.75);background-color:black;'><br>",
    
)
regressor = gr.Interface(
    fn = run_two,
    inputs = gradient_input,
    outputs = gradient_output,  
    submit_btn = "Train Model 2",
    title="Train your own model!",description="<img src='https://i.ibb.co/Bw08434/logo-1.png' alt='Logo' style='width:230px;height:100px;border-radius:5px;box-shadow:2px 2px 5px 0px rgba(0,0,0,0.75);background-color:black;'><br>",
    
)
train = gr.Interface(
    fn = predict_insurance_cost,
    inputs = inp,
    outputs = output, 
    submit_btn="Predict",
    title="Predict Health Insurance Cost!!",description="<img src='https://i.ibb.co/Bw08434/logo-1.png' alt='Logo' style='width:230px;height:100px;border-radius:5px;box-shadow:2px 2px 5px 0px rgba(0,0,0,0.75);background-color:black;'><br>Predict the insurance cost of an instance here!!",
   
)
demo = gr.TabbedInterface([random_forest, regressor, train], ["Train Model 1","Train Model 2", "Predict"])

if __name__ == "__main__":
    demo.launch()