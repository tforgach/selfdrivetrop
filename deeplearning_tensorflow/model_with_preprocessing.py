import tensorflow as tf
from tensorflow.keras import layers, datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import glob
from sklearn.model_selection import train_test_split

# preprocesses the images
def preprocessData(imgs, labels, imgHeight, imgWidth):
    processedImgs = []
    for path in imgs:
        img = cv.imread(path)
        img = cv.resize(img, (imgWidth, imgHeight))
        img = img.astype(np.float32) / 255.0 # normalizes the pixel values
        processedImgs.append(img)
    processedImgs = np.array(processedImgs)
    return processedImgs, labels 

# load data for preprocessing
imgHeight = 100
imgWidth = 100
imgDir = "DeepSteer/rosbags" # image directory
imgPaths = glob.glob(imgDir + "/*.jpg")

# load additional data needed for model
channels = 3
learning_rate = 0.01

# load datasets
# dsTrain = datasets.load('DeepSteer/rosbags/bag1.bag')
# dsTest = datasets.load('DeepSteer/rosbags/bag1.bag')
pathTrain, pathTest = train_test_split(imgPaths, test_size=0.2)
# xTrain = [x for x,y in dsTrain]
# xTest = [x for x,y in dsTest]
# yTrain = [y for x,y in dsTrain]
# yTest = [y for x,y in dsTest]
testSize = len(pathTrain)

# preprocess data
xTrain, yTrain = preprocessData(pathTrain, np.zeros(testSize), imgHeight, imgWidth)
xTest, yTest = preprocessData(xTest, yTest, np.zeros(testSize),imgHeight, imgWidth)


# define model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation=tf.nn.leaky_relu, inputShape=(imgHeight, imgWidth, channels)), 
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
        
