import gradio as gr
import joblib
import numpy as np

def predict_rentals(temp, humidity, windspeed, season, holiday, workingday, hour):
    # Load the pre-trained Random Forest model
    model = joblib.load("random_forest_model.pkl")
    
    # Prepare input features as a NumPy array
    features = np.array([[temp, humidity, windspeed, season, holiday, workingday, hour]])
    
    # Predict bike rental count
    prediction = model.predict(features)[0]
    return int(prediction)

# Define Gradio interface
interface = gr.Interface(
    fn=predict_rentals,
    inputs=[
        gr.Slider(0, 40, label="Temperature (°C)"),
        gr.Slider(0, 100, label="Humidity (%)"),
        gr.Slider(0, 50, label="Windspeed (km/h)"),
        gr.Dropdown([1, 2, 3, 4], label="Season (1: Spring, 2: Summer, 3: Fall, 4: Winter)"),
        gr.Radio([0, 1], label="Holiday (0: No, 1: Yes)"),
        gr.Radio([0, 1], label="Working Day (0: No, 1: Yes)"),
        gr.Slider(0, 23, step=1, label="Hour of the Day"),
    ],
    outputs="number",
    title="Bike Rental Prediction",
    description="Predict the number of bikes rented based on weather, time, and working conditions."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)