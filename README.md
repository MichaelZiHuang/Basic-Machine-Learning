# Basic-Machine-Learning
This is a "hello world" style machine learning project, focused on my first steps at learning machine learning and tensorflow

<h1>Day 1</h1>
<h2>Hello-World - Google Tutorial</h2>
This first tutorial is from the Tensorflow tutorial provided by google, we will go through this first before we
begin any image processing.
<pre><code>
import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
model.fit(xs, ys, epochs=500)
print(model.predict([10.0]))

def hello():
    print("hello world")

if __name__ == "__main__":
    hello()
</pre></code>

Let's look at this code. First, let's look at the dense layers
<code>model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])</code>
Let's break it down. "Sequential" means a linear set of layers, so if we add layer upon layer, it will go through them one by one.
It is worth noting that this does not require an initial layer, if we wanted to, we could do this
<code>
model = tf.keras.Sequential()
model.add(keras.layers.Dense(units=1, input_shape=[1]))
</code>
This is functionally similar, I advise doing this for clarity's sake. The "Unit" count is the number of neurons, a neuron
being a mathematical unit capable of taking inputs and outputting a single scalar value. We only have 1 since our input size is
1. This is enforced by the loss method
<code> model.compile(optimizer='sgd', loss='mean_squared_error') </code>
If we change the unit count to mismatch the input size, it will yell at us. 

<pre><code>
model.fit(xs, ys, epochs=500)
print(model.predict([10.0]))
</pre></code>
This above portion puts it all together. First, the "fit" portion will "train" our stuff 500 times, testing each of the inputs on each other.
<pre><code>
Epoch 6/500
6/6 [==============================] - 0s 165us/sample - loss: 33.9230
Epoch 7/500
6/6 [==============================] - 0s 167us/sample - loss: 27.8143
Epoch 8/500
6/6 [==============================] - 0s 334us/sample - loss: 23.0488
Epoch 9/500
6/6 [==============================] - 0s 334us/sample - loss: 19.3303
Epoch 10/500
6/6 [==============================] - 0s 334us/sample - loss: 16.4279
Epoch 11/500
6/6 [==============================] - 0s 334us/sample - loss: 14.1615
Epoch 12/500
6/6 [==============================] - 0s 167us/sample - loss: 12.3910
Epoch 13/500
6/6 [==============================] - 0s 167us/sample - loss: 11.0069
Epoch 14/500
6/6 [==============================] - 0s 167us/sample - loss: 9.9241
</pre></code>

Notice that it matches the 6 length of our arrays AND the loss continually decrease, we are getting more accurate as time goes on. 

Overall, this is a good mathematical tutorial, we will need to move on to image processing later. 

<h2>Cifar Tutorial</h2>
This is the Cifar tutorial I found, this uses pure keras instead of Tensorflow, this uses image processing to solve our problems. Notably, this uses the Cifar dataset, a pre-processed Keras dataset, we do NOT need to process our data, I think the next tutorial we will try to process some data for our future projects. Before I
close this one, I do want to see if I convert the file into tensorflow format. 

<pre><code>
#import tensorflow as tf
import numpy as np
#from tensorflow import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.datasets import cifar10
</pre></code>

Here's our import statement. Notice that I've included tensorflow. There is a noticeable conflict between TF and Keras, this stopped happening once I stopped the conflict.

<pre><code>
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0
</pre></code>


