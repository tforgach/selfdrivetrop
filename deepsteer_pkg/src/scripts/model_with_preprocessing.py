import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
from sklearn.model_selection import train_test_split

# load data for preprocessing
imgDir = "deepsteer_pkg/extracted_images" # image directory
imgPaths = [os.path.join(imgDir, filename) for filename in os.listdir(imgDir) if filename.endswith(".jpg")] # load data
labels = np.zeros(len(imgPaths)) # load data

# define image resize parameters
imgHeight = 100
imgWidth = 100

# preprocesses the images
def preprocessData(imgs, labels, imgHeight, imgWidth):
    processedImgs = []
    for path in imgs:
        img = cv.imread(path)
        img = cv.resize(img, (imgWidth, imgHeight))
        img = img.astype(np.float32) / 255.0 # normalizes the pixel values
        processedImgs.append(img)
    processedImgs = np.array(processedImgs)
    return processedImgs #, labels - may not need preprocessing

# load additional data needed for model
channels = 3
learning_rate = 0.01

# define model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation=tf.nn.leaky_relu, input_shape=(imgHeight, imgWidth, channels)), 
    layers.MaxPooling2D((2,2)), 
    layers.Conv2D(64, (3, 3), activation=tf.nn.leaky_relu), 
    layers.MaxPooling2D((2,2)), 
    layers.Conv2D(64, (3, 3), activation=tf.nn.leaky_relu), 
    layers.Flatten(), 
    layers.Dense(64, activation=tf.nn.leaky_relu), 
    layers.Dense(1)
])

opt = tf.keras.optimizers.SGD(learning_rate = learning_rate)

model.compile(optimizer=opt, loss="mse", metrics=["mae"])

# split and train model
xTrain, xTest, yTrain, yTest = train_test_split(imgPaths, labels, test_size=0.2, random_state=42)

# preprocess data
xTrainProcessed = preprocessData(xTrain, imgHeight, imgWidth)
xTestProcessed = preprocessData(xTest, imgHeight, imgWidth)
testSize = len(xTestProcessed)

model.fit(xTrainProcessed, yTrain, epochs=10, batch_size = 1) # train model
model.save("trained_model") # saves the trained model
print("Successfully saved")
model = tf.keras.models.load_model("trained_model")

menu = input(f"Training is complete, moving on to tests with {testSize} feeatures.\n\
1: Display each prediction and actual value\n\
2: Display each prediction, actual value and the input image\n\
3: Display accuracy plot after tests (Both of the other options will do this as well.)\n")
if menu == "1":
    predictions = model.predict(xTestProcessed)
    for i in range(testSize):
        pred = predictions[i][0]
        y = yTest[i][0]
        input(f'Feature {i}\nModel predicted: {pred}. Actual label: {y}\nLoss of: {abs(pred-y)}')

elif menu == "2":
    for i in range(testSize):
        x = np.expand_dims(xTestProcessed[i], axis=0)
        y = yTest[i][0]
        img = xTestProcessed[i].np()
        pred = model.pred(x)[0][0]
        cv.imshow("Camera View", img)
        print(f'Feature {i}\nModel predicted: {pred}. Actual label: {y}\nLoss of: {abs(pred-y)}')
    cv.waitKey(0)

xTestProcessed = tf.convert_to_tensor(xTestProcessed)
yTest = tf.convert_to_tensor(yTest)
testPerf = model.evaluate(xTestProcessed, yTest, batch_size = 1, verbose=0) #test performance

plt.plot(testPerf.history['loss'], label='loss')
plt.plot(testPerf.history['mae'], label='mae')
plt.ylabel('loss')
plt.legend()
plt.show()
