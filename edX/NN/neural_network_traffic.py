

import cv2
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    
    model = get_model()

    
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    
    images = list()
    labels = list()

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            print(f"Loading files from {folder_path}...")
            for file in os.listdir(folder_path):
                try:
                    image = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_COLOR)
                    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
                    images.append(image)
                    labels.append(int(folder))
                except Exception as e:
                    print(f"There is a problem with file: {file}")
                    print(str(e))

    return images, labels


def get_model():
    
    model = tf.keras.models.Sequential([

        
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # Max-pooling layer, using 3x3 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

        # Flatten units
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(NUM_CATEGORIES * 32, activation="relu"),
        tf.keras.layers.Dropout(0.2),

        # Add a hidden layer
        tf.keras.layers.Dense(NUM_CATEGORIES * 16, activation="relu"),

        # Add a hidden layer
        tf.keras.layers.Dense(NUM_CATEGORIES * 8, activation="relu"),

        # Add an output layer with output units for all categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
