import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from scipy import sparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
from tensorflow.keras.models import save_model, load_model
from imgaug import augmenters as iaa
import dask.array as da
import csv
from collections import Counter


def preprocessv2(image):
    image = cv2.imread(image)
    image = image[80:220, 150:320, :]
    image = image / 255
    return image


def imageandlights(data):
    image_processed = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center = indexed_data[0]
        image_processed.append(center.strip())
        steering.append(indexed_data[1])
    image_paths = np.asarray(image_processed)
    steerings = np.asarray(steering)
    return image_paths, steerings


def imageandsteering(data):
    image_processed = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center = indexed_data[0]
        image_processed.append(center.strip())
        steering.append(float(indexed_data[1]))
    image_paths = np.asarray(image_processed)
    steerings = np.asarray(steering)
    return image_paths, steerings


def preprocess(image):
    image = mpimg.imread(image)
    image = image[100:200, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image / 255
    return image


def zoom(image):
    zoom = iaa.Affine(scale=(1, 1.3))
    image = zoom.augment_image(image)
    return image


def pan(image):
    pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image


def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image


def img_random_flip(image, steering_angle):
    image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle


def random_augment(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = img_random_flip(image, steering_angle)

    return image, steering_angle


def batch_generator(image_paths, steering_ang, batch_size, istraining):
    while True:
        batch_img = []
        batch_steering = []

        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)

            if istraining:
                im, steering = random_augment(image_paths[random_index], steering_ang[random_index])

            else:
                im = mpimg.imread(image_paths[random_index])
                steering = steering_ang[random_index]

            im = preprocess(im)
            batch_img.append(im)
            batch_steering.append(steering)
        yield (np.asarray(batch_img), np.asarray(batch_steering))


'''

def carla_preprocess(image):
	image=mpimg.imread(image)
	image=image[200:495,:,:]
	image=cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
	image=cv2.GaussianBlur(image,(3,3),0)
	image=cv2.resize(image,(200,66))
	image=image.reshape(1,66,200,3)


	return image


def preprocess(image):
	image=mpimg.imread(image)
	image=image[55:135,:,:]
	image=cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
	image=cv2.GaussianBlur(image,(3,3),0)
	image=cv2.resize(image,(200,66))

	image=image/255

	return image


def preprocessv2(image):
	image=image[9:24,:]
	image=cv2.resize(image,(80,60))
	#image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	image=cv2.GaussianBlur(image,(3,3),0)
	return image

def carla_preprocessv2(image):
	image=image[9:24,:]
	image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	image=cv2.GaussianBlur(image,(3,3),0)
	image=cv2.resize(image,(80,60))
	image=image.reshape(1,80,60,1)
	return image


def newpreprocess(image):
	#image=image[20:48,:]
	image=cv2.GaussianBlur(image,(3,3),0)
	return image
def preprocess(image):
	image=image.reshape(60,80,1)
	image=image/255	
	return image
'''


def createnvdia():
    model = Sequential()
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    # model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu'))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1))

    optimizer = Adam(lr=1e-4)
    model.compile(loss='mse', optimizer=optimizer)
    return model


def alexnet(width, height, lr):
    network = input_data(shape=[None, width, height, 1], name='input')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.7)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.7)
    network = fully_connected(network, 3, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=lr, name='targets')

    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=0, tensorboard_dir='log')

    return model


# LENET CNN ----START
def createLenet():
    model = keras.Sequential()

    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(140, 170, 3)))
    model.add(MaxPooling2D())

    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=4, activation='softmax'))

    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# LENET CNN---------END
def createdata():
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
    train_data = pd.read_csv("D:\w\DATA\driving_log.csv", names=columns)
    num_bins = 25
    samples_per_bin = 200
    hist, bins = np.histogram(train_data['steering'], num_bins)
    center = (bins[:-1] + bins[1:]) * 0.5

    remove_list = []
    for j in range(num_bins):
        list_ = []
        for i in range(len(train_data['steering'])):
            if train_data['steering'][i] >= bins[j] and train_data['steering'][i] <= bins[j + 1]:
                list_.append(i)
        list_ = shuffle(list_)
        list_ = list_[samples_per_bin:]
        remove_list.extend(list_)
    train_data.drop(train_data.index[remove_list], inplace=True)

    imageinputs, steeringoutputs = imageandsteering(train_data)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(imageinputs, steeringoutputs, test_size=0.2, random_state=6)
    # preprocess(Xtrain[0])

    Xtrain = np.array(list(map(preprocess, Xtrain)))
    Xtest = np.array(list(map(preprocess, Xtest)))
    print(Xtrain.shape)

    model = createnvdia()
    model.fit(Xtrain, Ytrain, epochs=30, validation_data=(Xtest, Ytest), batch_size=100, verbose=1, shuffle=1)
    model.save('D:\WindowsNoEditor\PythonAPI\examples\save_model-udacitydata-nvidia-augmented\saved_model')


def creatergbdata():
    columns = ['imagepaths', 'steer']
    data = pd.read_csv('D:\WindowsNoEditor\PythonAPI\examples\steeringvalues.csv', names=columns)
    num_bins = 25
    samples_per_bin = 18000
    hist, bins = np.histogram(data['steer'], num_bins)
    center = (bins[:-1] + bins[1:]) * 0.5
    # plt.bar(center,hist,width=0.05)
    # plt.show()
    print("balancing data")
    remove_list = []
    for j in range(num_bins):
        list_ = []
        for i in range(len(data['steer'])):
            if data['steer'][i] >= bins[j] and data['steer'][i] <= bins[j + 1]:
                list_.append(i)
        list_ = shuffle(list_)
        list_ = list_[samples_per_bin:]
        remove_list.extend(list_)
    data.drop(data.index[remove_list], inplace=True)
    # print("remaining")
    # print(len(data))
    # hist, bins = np.histogram(data['steer'], num_bins)
    # plt.bar(center,hist,width=0.05)
    # plt.show()
    print("imageandsteering")
    imageinputs, steeringoutputs = imageandsteering(data)

    print("dask arrays and processing")

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(imageinputs, steeringoutputs, test_size=0.2, random_state=6)
    # print(preprocess(Xtrain[0]))

    Xtrain = np.array(list(map(preprocess, Xtrain)), dtype='float16')
    Xtest = np.array(list(map(preprocess, Xtest)), dtype='float16')
    print(Xtrain[0])
    print(Xtrain.shape)

    print("started training")
    model = createnvdia()
    # history = model.fit_generator(batch_generator(Xtrain, Ytrain, 100, 1),steps_per_epoch=300, epochs=10,validation_data=batch_generator(Xtest, Ytest, 100, 0),validation_steps=200,verbose=1,shuffle = 1)
    history = model.fit(Xtrain, Ytrain, epochs=15, validation_data=(Xtest, Ytest), batch_size=100, verbose=1, shuffle=1)
    # model.save('D:\WindowsNoEditor\PythonAPI\examples\save_model-rgbdata-nvdia-1output-balanced-augmented\saved_model')

    loss_train1 = history.history['loss']
    loss_val1 = history.history['val_loss']
    epochs = range(1, 16)
    plt.plot(epochs, loss_train1, 'g', label='Training loss')
    plt.plot(epochs, loss_val1, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def testprediction():
    model = load_model('D:\WindowsNoEditor\PythonAPI\examples\save_model-trafficdata-4output\saved_model')
    image = 'D:/WindowsNoEditor/PythonAPI/examples/out00056129.jpg'
    image = mpimg.imread(image)
    image = image[80:220, 150:320, :]
    image = image / 255
    plt.imshow(image)
    plt.show()
    image = np.array([image])
    print(image.shape)

    prediction = model.predict(image)[0]
    print(np.around(prediction))

    '''
	checkingimage=cv2.cvtColor(checkingimage,cv2.COLOR_RGB2GRAY)
	checkingimage=checkingimage[9:24,:]
	checkingimage=cv2.resize(checkingimage,(80,60))
	checkingimage=checkingimage.reshape(1,60,80,1)
	checkingimage=checkingimage/255
	
	prediction=model.predict(checkingimage)[0]
	print(prediction)
'''


def trainalexnetwithmanualdata():
    train_data = np.load('train_data_balanced.npy', encoding='latin1', allow_pickle=True)
    train = train_data[:-100]
    test = train_data[-100:]
    Xtrain = np.array([preprocess(i[0]) for i in train])
    Ytrain = np.array([i[1] for i in train])

    test_x = np.array([preprocess(i[0]) for i in test])
    test_y = np.array([i[1] for i in test])
    print(Xtrain.shape)

    # model=createLenet()
    model = alexnet(60, 80, 1e-3)
    model.fit(Xtrain, Ytrain, n_epoch=11, validation_set=(test_x, test_y), snapshot_step=500, show_metric=True)
    # model.fit(Xtrain,Ytrain,epochs=10,validation_split=0.2,batch_size=400,verbose=1,shuffle=1)
    save_model(model, 'D:\WindowsNoEditor\PythonAPI\examples\save_model-manualdata-alexnet-crop-3output\saved_model')


def trainlenetwithmanualdata():
    train_data = np.load('train_data_balanced_balanced.npy', encoding='latin1', allow_pickle=True)
    Xtrain = np.array([preprocess(i[0]) for i in train_data])
    Ytrain = np.array([i[1] for i in train_data])
    print(Xtrain.shape)

    model = createLenet()
    model.fit(Xtrain, Ytrain, epochs=11, validation_split=0.2, batch_size=400, verbose=1, shuffle=1)
    save_model(model,
               'D:\WindowsNoEditor\PythonAPI\examples\save_model-manualdata-lenet-nocrop-3output-nonbogus\saved_model')


def trainnvdiawithmanualdata():
    train_data = np.load('train_data_balanced.npy', encoding='latin1', allow_pickle=True)
    train = train_data[:-100]
    test = train_data[-100:]
    # Xtrain=np.array([preprocess(i[0]) for i in train])
    Ytrain = np.array([i[1] for i in train])

    # test_x=np.array([preprocess(i[0]) for i in test])
    test_y = np.array([i[1] for i in test])

    print(Ytrain.shape)
    '''
	model=createnvdia()
	model.fit(Xtrain,Ytrain,epochs=30,validation_data=(test_x,test_y),batch_size=100,verbose=1,shuffle=1)
	model.save('D:\WindowsNoEditor\PythonAPI\examples\save_model-manualdata-alexnet-crop-3output\saved_model')
'''


def preprocessv22(label):
    if label == '[1 0 0 0]':
        return np.array([1, 0, 0, 0])
    elif label == '[0 1 0 0]':
        return np.array([0, 1, 0, 0])
    elif label == '[0 0 1 0]':
        return np.array([0, 0, 1, 0])
    else:
        return np.array([0, 0, 0, 1])


def create_trafficdata():
    columns = ['imagepaths', 'imagesize', 'fileattribute', 'regioncount', 'regionid', 'regionshape', 'type', 'light']
    data = pd.read_csv('D:/WindowsNoEditor/PythonAPI/examples/trafficlightsv2.csv', names=columns)

    print("imageandlights")
    imageinputs, lightoutputs = imageandlights(data)
    letter_counts = Counter(lightoutputs)
    df = pd.DataFrame.from_dict(letter_counts, orient='index')
    df.plot(kind='bar')
    plt.show()
    num_bins = 2
    samples_per_bin = 800

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(imageinputs, lightoutputs, test_size=0.2, random_state=6)

    Xtrain = np.array(list(map(preprocessv2, Xtrain)))
    Xtest = np.array(list(map(preprocessv2, Xtest)))
    Ytrain = np.array(list(map(preprocessv22, Ytrain)))
    Ytest = np.array(list(map(preprocessv22, Ytest)))

    print(Xtrain[80])
    print(Xtrain.shape)
    print(Ytrain[80])
    print(Ytrain.shape)


'''
	print("started training")
	model=createLenet()
	model.fit(Xtrain,Ytrain,epochs=10,validation_data=(Xtest,Ytest),batch_size=20,verbose=1,shuffle=1)
	save_model(model,'D:\WindowsNoEditor\PythonAPI\examples\save_model-trafficdata\saved_model')

'''


def create_trafficdatav2():
    columns = ['imagepaths', 'light']
    data = pd.read_csv('D:/WindowsNoEditor/PythonAPI/examples/trafficfinal.csv', names=columns)

    print("imageandlights")
    imageinputs, lightoutputs = imageandlights(data)
    # letter_counts = Counter(lightoutputs)
    # df = pd.DataFrame.from_dict(letter_counts, orient='index')
    # df.plot(kind='bar')
    # plt.show()
    # num_bins=2
    # samples_per_bin=800
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(imageinputs, lightoutputs, test_size=0.25, random_state=6)

    Xtrain = np.array(list(map(preprocessv2, Xtrain)))
    Xtest = np.array(list(map(preprocessv2, Xtest)))
    Ytrain = np.array(list(map(preprocessv22, Ytrain)))
    Ytest = np.array(list(map(preprocessv22, Ytest)))
    # plt.imshow(Xtrain[100])
    # plt.show()
    # print(Xtrain.shape)
    # print(Ytrain[100])
    # print(type(Ytrain[100]))
    # print(Ytrain.shape)

    print("started training")
    model = createLenet()
    history = model.fit(Xtrain, Ytrain, epochs=8, validation_data=(Xtest, Ytest), batch_size=100, verbose=1, shuffle=1)
    save_model(model, 'D:\WindowsNoEditor\PythonAPI\examples\save_model-trafficdata-4output-pbfile\saved_model.pb')
    loss_train = history.history['acc']
    loss_val = history.history['val_acc']
    epochs = range(1, 9)
    plt.plot(epochs, loss_train, 'g', label='Training accuracy')
    plt.plot(epochs, loss_val, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    loss_train1 = history.history['loss']
    loss_val1 = history.history['val_loss']
    epochs = range(1, 9)
    plt.plot(epochs, loss_train1, 'g', label='Training loss')
    plt.plot(epochs, loss_val1, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def convertmodel():
    model = load_model('D:\WindowsNoEditor\PythonAPI\examples\save_model-rgbdata-nvdia-1output-balanced\saved_model')
    save_model(model, 'D:\WindowsNoEditor\PythonAPI\examples\save_model-manualdata-pbfile\saved_model.pb')


if __name__ == "__main__":
    creatergbdata()

# testprediction(image)

# create the CNN model
# model=createLenet()
# trainmodel(model)
# model.save('D:\WindowsNoEditor\PythonAPI\examples')
