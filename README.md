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

This part is a bit rough to understand. First, let's talk about the first two lines. First, it turnns the X sets into float32 point, I believe this is done for sizes and ease of math since initiailly, the are integers
Then, it divides it by 255.0. This is effectively one-hot encoding. One-hot is a way ML algorithms to easily give something a categorical value. An example would be assigning a bunch of car companies, each would have a different categorical value.

<pre><code>
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
class_num = Y_test.shape[1]
</pre></code>

Here's the same thing for the Y setup. I should note the "class num part". Essentially, it denotes how many neurons it shouild compress down to at the final layer. In other words, how the neuron should take its input and transform its output. 

<pre><code>
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], activation='relu',padding='same'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
</pre></code>

Probably one of the most interesting parts. Let's start with the "Sequential" line. It's, surprisingly, straight forward. Essentially, the sequential model is exactly what it sounds like, it will linearly go through layers to achieve its goal. 

Later, we add the "Conv2d" layer. This is called the convolutional layer. The convolution layer essentially takes your pre-processed data and assigns various levels of importance of bias to each section. This is called "feature maps", it takes the pixels of your image by taking several different filters, then relating similar parts to another. The activation function, essentially, is a way for us to get an output of a node. 

Dropout, quite literally, drops out 20% of the data we recieve. 

Batch Normalization is fairly straight fowrad. It tries to fight against "internal covariate shift". In other words, data likes to work with consistent data, but the inputs tend to be of different distributions, this is a good way to force inputs to have about the same distribution each time. In other words, let's say X had some maping to Y and it learned some distribution and algorithm. Then, X gets new inputs and its distribution changes, then the learning algorithm must retrain itself with the new distributions. 

Here's the best explanation for Batch NM I can give.
First, it normalizes the data by subtracting the batch mean and dividing by its standard deviation
Then, it adds two trainable parameters, a trainable mean and standard deivation. These are the weights that let us remove "denormalization" which our SGD works with. 

<pre><code>
model.add(MaxPooling2D(pool_size=(2, 2)))
</pre></code>

This final bit adds "pooling". This, essentially, takes our data and attempts to take only the most relevant pieces while throwing away the nonnecessary. This helps with overfitting.

To be completely honest, I am going to stop this program for now. I don't truely understand this, let's return to this later.