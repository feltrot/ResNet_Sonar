
# This script creates a deep neural network to classify objects from sonar images
# into marine life classes


# since this is the first deep neural network (DNN) I am planning to build, let's
# familiarise ourselves with some basic Machine Learning code from
# Google Colab (https://colab.research.google.com/notebooks/gpu.ipynb#scrollTo=sXnDmXR7RDr2)

# to compare the difference in computing speed between CPUs and GPUs
# the following example constructs a typical convolutional neural network layer 
# over a random image and manually places the resulting ops on 
# either the CPU or the GPU to compare execution speed.

# load modules
import tensorflow as tf
import timeit

# names the device being used for the analyses
device_name = tf.test.gpu_device_name() 

# test if the device name is set to a GPU. If not, raise an error
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
# if the device name is set correctly to a GPU, print the device name
# according to it's format
print('Found GPU at: {}'.format(device_name))

# get the device name
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise SystemError('GPU device not found')

# create functions that creates a random image and then constructs a
# convolutional neural network (CNN) over the random image with either CPU or GPU
def cpu():
  with tf.device('/cpu:0'):
    random_image_cpu = tf.random.normal((100, 100, 100, 3))
    net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
    return tf.math.reduce_sum(net_cpu)

def gpu():
  with tf.device('/device:GPU:0'):
    random_image_gpu = tf.random.normal((100, 100, 100, 3))
    net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
    return tf.math.reduce_sum(net_gpu)
  
# We run each op once to warm up; see: https://stackoverflow.com/a/45067900
cpu()
gpu()

# Now compare the two run speeds, and determine how much faster GPU is
# Run the op several times.
print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
      '(batch x height x width x channel). Sum of ten runs.')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))





###############################################################################
## Now lets try and build our first model

# load modules
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Function to enhance the quality of images
def enhance_image_quality(image):
    # Add your image enhancement code here
    # Example: Use OpenCV for resizing
    enhanced_image = cv2.resize(image, (224, 224))
    return enhanced_image

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
model = Model(inputs=base_model.input, outputs=x)

# Function to extract features using ResNet50
def extract_resnet_features(img):
    img = enhance_image_quality(img)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img)
    return features.flatten()

# Load time series images from locally saved files
image_folder = 'path_to_your_image_folder'
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Assuming labels are derived from file names or some other logic
labels = [int(filename.split('_')[0]) for filename in image_files]

# Load images and extract features
time_series_images = []
features = []

for filename in image_files:
    img_path = os.path.join(image_folder, filename)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ResNet50 expects RGB format
    time_series_images.append(img)
    
    img_features = extract_resnet_features(img)
    features.append(img_features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a simple classifier (replace with more complex models as needed)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

classifier = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1))
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Now you can use the classifier to predict labels for new images
# Example: new_img = cv2.imread('path_to_new_image.jpg')
# new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
# new_features = extract_resnet_features(new_img)
# prediction = classifier.predict([new_features])
# print(f"Predicted label: {prediction}")
