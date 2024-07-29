import tensorflow as tf



# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def get_training_data():
    # # Print the first 10 instances
    # for i in range(10):
    #     print(f"Instance {i+1}:")
    #     print(x_train[i])
    #     print()
    return x_train, x_test 


class Model:
    def __init__(self):
        # Create a simple model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
            ])

    def compile(self):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        adam = tf.keras.optimizers.Adam()
        self.model.compile(optimizer=adam,
              loss=loss_fn,
              metrics=['accuracy'])

    def train(self, input_train, output_train): # Train the model
        history = self.model.fit(input_train, output_train, epochs=5)

    def save(self):
        self.model.save('mnist.keras')

    def evaluate(self, x_test,  y_test, verbose=2):
        self.model.evaluate(x_test,  y_test, verbose=verbose)

    print("Done")

def main():
    model = Model()
    model.compile()
    model.train(x_train, y_train)
    model.save()
    model.evaluate(x_test,  y_test, verbose=2)

    print("Done")

if __name__ == "__main__":
    main()