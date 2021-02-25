# AlexNet Implementation

# AlexNet implementation using the CIFAR10 dataset. 
# Paper can be found at: [https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf]


# Imports
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time

# # Using CIFAR10 dataset from Keras
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Classes in dataset (10 classes)
CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Creating validation set -> last 5000 images of training dataset
validation_images, validation_labels = train_images[:5000], train_labels[:5000]
train_images, train_labels = train_images[5000:], train_labels[5000:]


# Data Preprocessing

# Converting dataset object to tensorflow dataset object for implementation of pipelines
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

# Visualizing CIFAR dataset
plt.figure(figsize=(20,20))
for i, (image, label) in enumerate(train_ds.take(5)):
    ax = plt.subplot(5,5,i+1)
    plt.imshow(image)
    plt.title(CLASS_NAMES[label.numpy()[0]])
    plt.axis('off')


# Normalizing and Resizing images
def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image, label


# Data Pipeline

train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
print("Training data size:", train_ds_size)
print("Test data size:", test_ds_size)
print("Validation data size:", validation_ds_size)

# 2 operations:
# 1. Preprocessing the data within the dataset
# 3. Batch data within the dataset
train_ds = (train_ds
                  .map(process_images)
                  .batch(batch_size=128, drop_remainder=True))
test_ds = (test_ds
                  .map(process_images)
                  .batch(batch_size=32, drop_remainder=True))
validation_ds = (validation_ds
                  .map(process_images)
                  .batch(batch_size=32, drop_remainder=True))


# Model Implementation
model = keras.models.Sequential([
    # Input layer
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', 
                        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                        input_shape=(227,227,3), use_bias = True,  bias_initializer='zeros'),
    tf.keras.layers.Lambda(tf.nn.local_response_normalization,
                           {
                               "depth_radius": 5.0,
                               "bias": 2.0,
                               "alpha": 0.0001,
                               "beta": 0.75
                           }
                          ),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    
    # Second layer
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu',
                        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                        padding="same", use_bias = True,  bias_initializer='ones'),
    tf.keras.layers.Lambda(tf.nn.local_response_normalization,
                           {
                               "depth_radius": 5.0,
                               "bias": 2.0,
                               "alpha": 0.0001,
                               "beta": 0.75
                           }
                          ),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    
    # Third layer
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', 
                        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                        padding="same", use_bias = True,  bias_initializer='zeros'),
     tf.keras.layers.Lambda(tf.nn.local_response_normalization,
                           {
                               "depth_radius": 5.0,
                               "bias": 2.0,
                               "alpha": 0.0001,
                               "beta": 0.75
                           }
                          ),
    
    # Fourth layer
    keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', 
                        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                        padding="same", use_bias = True,  bias_initializer='ones'),
    tf.keras.layers.Lambda(tf.nn.local_response_normalization,
                           {
                               "depth_radius": 5.0,
                               "bias": 2.0,
                               "alpha": 0.0001,
                               "beta": 0.75
                           }
                          ),
    
    # Fifth layer
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', 
                        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                        padding="same", use_bias = True,  bias_initializer='ones'),
    tf.keras.layers.Lambda(tf.nn.local_response_normalization,
                           {
                               "depth_radius": 5.0,
                               "bias": 2.0,
                               "alpha": 0.0001,
                               "beta": 0.75
                           }
                          ),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    
    # Sixth layer
    keras.layers.Flatten(),
    
    keras.layers.Dense(4096, activation='relu', bias_initializer='ones',
                      kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),),
    keras.layers.Dropout(0.5),
    
    # Seventh layer
    keras.layers.Dense(4096, activation='relu', bias_initializer='ones',
                      kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),),
    keras.layers.Dropout(0.5),
    
    # Output layer
    keras.layers.Dense(10, activation='softmax',
                      kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),)
])


# Reduce learning rate
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                 factor=0.1, patience=1,
                                                 min_lr=0.00001)


# Training
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
model.summary()

model.fit(train_ds,
          epochs=90,
          validation_data=validation_ds,
          callbacks=[reduce_lr],
          batch_size=128)