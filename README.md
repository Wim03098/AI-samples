TensorFlow Number Size Classification
This project uses TensorFlow to classify numbers into three categories: “small”, “medium”, and “big”. The classification is based on predefined thresholds.

Dependencies
TensorFlow
NumPy
Python’s built-in random and datetime modules
Constants
The Constants class defines several parameters used in the project:

TRAINING_RECORDS: The number of training records.
EPOCHS: The number of epochs for training the model.
SMALL_THRESHOLD: The upper limit for a number to be considered “small”.
MEDIUM_THRESHOLD: The upper limit for a number to be considered “medium”.
BIG_THRESHOLD: The upper limit for a number to be considered “big”.
SIZE: The size of the training data.
Training Data
The get_training_data function generates the training data. It creates a list of random integers and classifies them into “small”, “medium”, or “big” based on the thresholds defined in the Constants class.

Model
The Model class defines a simple TensorFlow model with two layers:

A dense layer with one unit and an input shape of one.
A dense layer with three units and a softmax activation function.
The model is compiled with the SGD optimizer, sparse categorical crossentropy loss, and accuracy metric. The train method trains the model, and the save method saves the model to a file named ‘number_size.keras’.

Main Function
The main function creates an instance of the Model class, compiles the model, generates the training data, trains the model, and saves the model.

Usage
To run the script, use the following command:

python script_name.py

Replace script_name.py with the name of your Python script.

Output
The script prints the training data and a “Done” message when it finishes running. The trained model is saved to a file named ‘number_size.keras’.
