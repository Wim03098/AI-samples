import tensorflow as tf
import random
import numpy as np

SIZE = 3000.0

# Later, you can load the model like this:
loaded_model = tf.keras.models.load_model('my_model.keras')
print(loaded_model.summary())

# Now you can use the loaded model to make predictions
def predict_with_loaded_model(numbers):       
    numbers_array = np.array(numbers)
    normalised_numbers = numbers_array / SIZE    
# Sort the array in ascending order   
    print(normalised_numbers)
    predictions = loaded_model.predict(normalised_numbers)
    return predictions

# Create some example data
numbers = []
# for i in range(50):
#     numbers.append(random.randint(1, SIZE)) 

for i in range(-1000, 4000, 100):
    numbers.append(i) 
numbers = sorted(numbers)
print(numbers)
predictions = predict_with_loaded_model(numbers)  # Should print ["small", "big"]

for num, pred in zip(numbers, predictions):
    # Get the index of the maximum value (0 for "small", 1 for "big")
    predicted_class = np.argmax(pred)
    
    # Convert probabilities to percentages and round to 0 decimals
    pred_percentages = np.round(pred * 100, 0).astype(int)

    # Map the predicted class to the corresponding label
    labels = ["small", "medium", "big"]    
    print(f"Number: {int(num)}, Prediction: {pred_percentages}%, {labels[predicted_class]}")