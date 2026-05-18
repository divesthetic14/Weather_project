from tensorflow.keras.models import load_model

model = load_model("models/forecasting_gru_tuned_model.h5", compile=False)

model.save("models/forecasting_gru_new_tuned_model.keras")