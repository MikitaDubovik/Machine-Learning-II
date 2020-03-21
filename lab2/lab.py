import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import lab_common as lab

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def train(x_train, x_test, y_train, y_test, epochs,img_height,img_width):
    model=tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(img_height, img_width)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=epochs)
    model.evaluate(x_test, y_test)

def train_with_reg(x_train, x_test, y_train, y_test, epochs,img_height,img_width):
    model=tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(img_height, img_width)),
        tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=epochs)
    model.evaluate(x_test, y_test)

def run():
    dataset_url="https://commondatastorage.googleapis.com/books1000/notMNIST_large.tar.gz"
    img_height=img_width=28
    classes=["A","B","C","D","E","F","G","H","I","J"]
    epochs=100

    dataset_root_path=lab.extract_dataset(lab.download_dataset(dataset_url))
    hdf5_path=lab.create_hdf5(dataset_root_path,classes,img_height,img_width)
    X,Y=lab.read_hdf5(hdf5_path)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.04, random_state=42)
    train(X_train, X_test, Y_train, Y_test, epochs,img_height,img_width)
    train_with_reg(X_train, X_test, Y_train, Y_test, epochs,img_height,img_width)
    

if __name__ == "__main__":
    run()