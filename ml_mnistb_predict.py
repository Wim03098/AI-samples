import tensorflow as tf
import random
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Later, you can load the model like this:
loaded_model = tf.keras.models.load_model('mnist.keras')
print(loaded_model.summary())

probability_model = tf.keras.Sequential([
  loaded_model,
  tf.keras.layers.Softmax()
])

predictions = probability_model.predict(x_test)

for num, pred in zip(x_test, predictions):
    # Get the index of the maximum value (0 for "small", 1 for "big")
    predicted_class = np.argmax(pred)
    
    # Convert probabilities to percentages and round to 0 decimals
    pred_percentages = np.round(pred * 100, 0).astype(int)

    # Map the predicted class to the corresponding label
    labels = ["zero", "one", "two", "3", "4", "5", "6", "7", "8", "9"]    
    print(f"Number:, Prediction: {pred_percentages}%, {labels[predicted_class]}")


print("done")