import gradio as gr
import joblib
import numpy as np

def predict_rentals(season, hr, holiday, workingday, weathersit, temp, atemp, hum, windspeed,
                     yr, mnth, weekday_Fri, weekday_Mon, weekday_Sat, weekday_Sun, 
                     weekday_Thu, weekday_Tue, weekday_Wed):
    # Load the pre-trained Random Forest model
    model = joblib.load("random_forest_model.pkl")
    
    # Prepare input features as a NumPy array
    features = np.array([[season, hr, holiday, workingday, weathersit, temp, atemp, hum, windspeed,
                           yr, mnth, weekday_Fri, weekday_Mon, weekday_Sat, weekday_Sun, 
                           weekday_Thu, weekday_Tue, weekday_Wed]])
    
    # Predict bike rental count
    prediction = model.predict(features)[0]
    return int(prediction)

# Define Gradio interface
interface = gr.Interface(
    fn=predict_rentals,
    inputs=[
        gr.Dropdown([1, 2, 3, 4], label="Season (1: Spring, 2: Summer, 3: Fall, 4: Winter)"),
        gr.Slider(0, 23, step=1, label="Hour of the Day"),
        gr.Radio([0, 1], label="Holiday (0: No, 1: Yes)"),
        gr.Radio([0, 1], label="Working Day (0: No, 1: Yes)"),
        gr.Dropdown([1, 2, 3, 4], label="Weather Situation (1: Clear, 2: Mist, 3: Light Rain, 4: Heavy Rain)"),
        gr.Slider(0, 40, label="Temperature (°C)"),
        gr.Slider(0, 50, label="Apparent Temperature (°C)"),
        gr.Slider(0, 100, label="Humidity (%)"),
        gr.Slider(0, 50, label="Windspeed (km/h)"),
        gr.Radio([0, 1], label="Year (0: 2011, 1: 2012)"),
        gr.Slider(1, 12, step=1, label="Month"),
        gr.Radio([0, 1], label="Friday"),
        gr.Radio([0, 1], label="Monday"),
        gr.Radio([0, 1], label="Saturday"),
        gr.Radio([0, 1], label="Sunday"),
        gr.Radio([0, 1], label="Thursday"),
        gr.Radio([0, 1], label="Tuesday"),
        gr.Radio([0, 1], label="Wednesday"),
    ],
    outputs="number",
    title="Bike Rental Prediction",
    description="Predict the number of bikes rented based on weather, time, and working conditions."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
