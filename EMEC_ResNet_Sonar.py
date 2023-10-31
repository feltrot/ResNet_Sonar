
# This script creates a deep neural network to classify objects from sonar images
# into marine life classes


# since this is the first deep neural network (DNN) I am planning to build, let's
# familiarise ourselves with some basic Machine Learning code from
# Google Colab (https://colab.research.google.com/notebooks/gpu.ipynb#scrollTo=sXnDmXR7RDr2)


import tensorflow as tf # imports the tensorflow package

# names the device being used for the analyses
device_name = tf.test.gpu_device_name() 

# test if the device name is set to a GPU. If not, raise an error
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
# if the device name is set correctly to a GPU, print the device name
# according to it's format
print('Found GPU at: {}'.format(device_name))


# to compare the difference in computing speed between CPUs and GPUs
# the following example constructs a typical convolutional neural network layer 
# over a random image and manually places the resulting ops on 
# either the CPU or the GPU to compare execution speed.

# load modules
import tensorflow as tf
import timeit

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

