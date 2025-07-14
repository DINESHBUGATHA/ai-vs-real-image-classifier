import tensorflow as tf

# Load your model
model = tf.keras.models.load_model('ai_vs_real_classifier.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Save the TFLite model
tflite_model = converter.convert()
with open('ai_vs_real_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model successfully converted to ai_vs_real_classifier.tflite")
