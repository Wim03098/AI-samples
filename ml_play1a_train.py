import tensorflow as tf
# from tensorflow.keras.callbacks import TensorBoard
import datetime
import numpy as np
import random

class Constants:
    # Define your "big" and "small" threshold
    TRAINING_RECORDS = 1000
    EPOCHS = 500
    SMALL_THRESHOLD = 1000
    MEDIUM_THRESHOLD = 2000
    BIG_THRESHOLD = 3000
    SIZE = 3000

def get_training_data():
    x_train = []
    y_train = []

    # Create some example data
    for i in range(Constants.SIZE):
        rnd = random.randint(1, Constants.SIZE)
        x_train.append(rnd)
        if rnd < Constants.SMALL_THRESHOLD:
            y_train.append(0)
        elif rnd < Constants.MEDIUM_THRESHOLD:
            y_train.append(1)        
        else: 
            y_train.append(2)
    print("x:", x_train, "y:", y_train)

    input_train = np.array(x_train, dtype=float)
    input_train = input_train / float(Constants.SIZE)
    output_train = np.array(y_train, dtype=int)  # 0 for "small", 1 for "big"

    return input_train, output_train

class Model:
    def __init__(self):
        # Create a simple model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=1, input_shape=[1]),
            tf.keras.layers.Dense(units=3, activation='softmax')
            ])

    def compile(self):
        sgd = tf.keras.optimizers.SGD(learning_rate=0.1)
        self.model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, input_train, output_train): # Train the model
        history = self.model.fit(input_train, output_train, epochs=20)

    def save(self):
        self.model.save('number_size.keras')

def main():
    model = Model()
    model.compile()
    input_train, output_train = get_training_data()
    model.train(input_train, output_train)
    model.save()
    print("Done")

if __name__ == "__main__":
    main()

