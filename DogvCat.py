from matplotlib import pyplot
from matplotlib.image import imread
from os import listdir
from os import makedirs
from os.path import join
from shutil import copyfile, copy2, copy
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from random import seed
from random import random

from keras.models import Sequential, model_from_json
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD

import sys



#direct = "datasets_dogs_vs_cats"
#subdirs = ['train/', 'test/'] 
#for subdir in subdirs:
#   labeldirs = ['dogs/', 'cats/']
#    for labeldir in labeldirs:
#        newdir = folder + subdir + labeldir
#        makedirs(newdir, exist_ok=True)

def Processing(p, l): #Uses a ton of ram
    classnum = 0 # 0 For dog, 1 for cat
    for file in listdir(folder):
        classnum = 0
        if file.startswith("cat"):
            classnum = 1
        #photo = load_img(folder + file, target_size=(200, 200))
        #photo = img_to_array(photo)
        #p.append(photo)
        l.append(classnum)
    #pyplot.show()
   # p = np.asarray(p)
    l = np.asarray(l)
    print(l.shape)
    #np.save('dogs_vs_cats_photos.npy', p)
    np.save('dogs_vs_cats_labels.npy', l)

#photos, labels = list(), list()

#photos = np.load('dogs_vs_cats_photos.npy')
#print(photos.shape)

#Processing(photos, labels)

def directProcessing(): # Correctly moves files to a different directory
    seed(1)
    val_ratio = .25
    path = "C:/Users/Michael Huang/Documents/GitHub/Basic-Machine-Learning/train/"
    dst_dir_base = "C:/Users/Michael Huang/Documents/GitHub/Basic-Machine-Learning/"
    for file in listdir(path):
        f = path + file
        dst_dir = "train/"
        if random() < val_ratio:
            dst_dir = "test/"
        if file.startswith('cat.'): # Important, if you do NOT denote cat., it will overlap with the cats/ folder
            des = dst_dir_base + dst_dir + 'cats/' + file
            copyfile(f, des)
        if file.startswith('dog.'):
            des = dst_dir_base + dst_dir + 'dogs/' + file
            copyfile(f, des)

def define_model():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu',  kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))
   

    model.add(Conv2D(32, (3, 3), activation='relu',  kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu',  kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu',  kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), activation='relu',  kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))


    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))
    #model.add(Dropout(0.2))

    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

def runMemes():
    model = define_model() # Gets our CNN, 1 layer model

    datagen = ImageDataGenerator(rescale=1.0/255.0)

    train = datagen.flow_from_directory(folder + 'train/', class_mode='binary', batch_size=64, target_size=(200, 200))# BIG NOTE, it is LOOKING for these sizes, that's why it's able to grab the dogs and cats folders.
    test = datagen.flow_from_directory(folder + 'test/', class_mode='binary', batch_size=64, target_size=(200, 200))
    history = model.fit_generator(train, steps_per_epoch=len(train), validation_data=test, validation_steps=len(test), epochs=50  , verbose=1)
    _, acc = model.evaluate_generator(test, steps=(len(test)), verbose=1)
    print('> %.3f' % (acc * 100.0))

    model_save = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_save)
    model.save_weights("model.hl5")
    summarize_diagnostics(history)



folder = "C:/Users/Michael Huang/Documents/GitHub/Basic-Machine-Learning/"

if __name__ == "__main__":
    runMemes()


