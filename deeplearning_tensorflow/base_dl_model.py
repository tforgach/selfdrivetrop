import tensorflow as tf
from tensorflow.keras import layers, datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

image_height = 100
image_width = 100
channels = 3
learning_rate = 0.01
image_dir = "bags/img/training"

dsTrain = datasets.load('tfds/tfds_train0000')
dsTest = datasets.load('tfds/tfds_test0000')
xTrain = [x for x,y in dsTrain]
xTest = [x for x,y in dsTest]
yTrain = [y for x,y in dsTrain]
yTest = [y for x,y in dsTest]

testSize = len(xTest)

model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation=tf.nn.leaky_relu, inputShape=(image_height, image_width, channels)), 
    layers.MaxPooling2D((2,2)), 
    layers.Conv2D(64, (3, 3), activation=tf.nn.leaky_relu), 
    layers.MaxPooling2D((2,2)), 
    layers.Conv2D(64, (3, 3), activation=tf.nn.leaky_relu), 
    layers.Flatten(), 
    layers.Dense(64, activation=tf.nn.leaky_relu), 
    layers.Dense(1)
])

opt = tf.keras.optimizers.SGD(learningRate = learning_rate)

model.compile(optimizer=opt, loss="mse", metrics=["mae"])
model.fit(xTrain, yTrain, epochs=10, batch_size = 1)

menu = input(f"Training is complete, moving on to tests with {testSize} feeatures.\n\
1: Display each prediction and actual value\n\
2: Display each prediction, actual value and the input image\n\
3: Display accuracy plot after tests (Both of the other options will do this as well.)\n")
if menu == "1":
    pred = model.predict(x)
    predictions = model.predict(xTest)
    for i in range(testSize):
        pred = predictions[i][0]
        y = yTest[i][0]
        input(f'Feature {i}\nModel predicted: {pred}. Actual label: {y}\nLoss of: {abs(pred-y)}')

elif menu == "2":
    for i in range(testSize):
        x = np.expand_dims(xTest[i], axis=0)
        y = yTest[i][0]
        img = xTest[i].np()
        pred = model.pred(x)[0][0]
        cv.imshow("Camera View", img)
        print(f'Feature {i}\nModel predicted: {pred}. Actual label: {y}\nLoss of: {abs(pred-y)}')
        
cv.waitKey(0)

xTest = tf.convert_to_tensor(xTest)
yTest = tf.convert_to_tensor(yTest)
testPerf = model.evaluate(xTest, yTest, batch_size = 1, verbose=0) #test performance

plt.plot(testPerf.history['loss'], label='loss')
plt.plot(testPerf.history['mae'], label='mae')
plt.ylabel('loss')
plt.legend()
plt.show()
        
