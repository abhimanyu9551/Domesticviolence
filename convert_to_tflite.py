import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("dv_model_v2.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save model
with open("dv_model_v2.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as dv_model_v2.tflite")
