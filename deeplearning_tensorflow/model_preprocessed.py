import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pandas as pd
from sklearn.model_selection import train_test_split
import rospy
from std_msgs.msg import Float32

# Load image paths and labels from the CSV file
data_df = pd.read_csv("") #CHANGE BASED ON YOUR COMPUTER, THIS THE FOLDER IMAGES ARE PUT INTO
imgPaths = data_df['image_path'].values
labels = data_df[['linear_x', 'angular_z']].values

# Define image resize parameters
imgHeight = 100
imgWidth = 100

def preprocessData(imgs, imgHeight, imgWidth):
    processedImgs = []
    for path in imgs:
        img = cv.imread(path)
        img = cv.resize(img, (imgWidth, imgHeight))
        img = img.astype(np.float32) / 255.0
        processedImgs.append(img)
    processedImgs = np.array(processedImgs)
    return processedImgs

# Load additional data needed for model
channels = 3
learning_rate = 0.01

# Define model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation=tf.nn.leaky_relu, input_shape=(imgHeight, imgWidth, channels)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation=tf.nn.leaky_relu),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation=tf.nn.leaky_relu),
    layers.Flatten(),
    layers.Dense(64, activation=tf.nn.leaky_relu),
    layers.Dense(2)
])

opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)

model.compile(optimizer=opt, loss="mse", metrics=["mae"])
#model.save_weights('model_weights.h5')
# Try to load model weights
try:
    # Try to load the model
    model = tf.keras.models.load_model("trained_model")
    print("Successfully loaded the model.")
except Exception as e:
    print("Could not load the model. Error: ", str(e))

# Split and train model
xTrain, xTest, yTrain, yTest = train_test_split(imgPaths, labels, test_size=0.05, random_state=42)

# Preprocess data
xTrainProcessed = preprocessData(xTrain, imgHeight, imgWidth)
xTestProcessed = preprocessData(xTest, imgHeight, imgWidth)
testSize = len(xTestProcessed)

history = model.fit(xTrainProcessed, yTrain, epochs=10, batch_size=1)

# Save model weights after training
model.save("trained_model")  # saves the trained model
print("Successfully saved")
model = tf.keras.models.load_model("trained_model")

menu = input(f"Training is complete, moving on to tests with {testSize} features.\n"
             f"1: Display each prediction and actual value\n"
             f"2: Display each prediction, actual value, and the input image\n"
             f"3: Display average centering error after tests as well as prediction, actual value, and input image\n")

if menu == "1":
    predictions = model.predict(xTestProcessed)
    for i in range(testSize):
        pred = predictions[i][0]
        y = yTest[i]
        print(f'Feature {i}\nModel predicted: {pred}. Actual label: {y}\nLoss of: {abs(pred - y)}')

elif menu == "2":
    for i in range(testSize):
        x = np.expand_dims(xTestProcessed[i], axis=0)
        y = yTest[i]
        img = xTestProcessed[i]
        pred = model.predict(x)[0]
        print(f'Feature {i}\nModel predicted: {pred}. Actual label: {y}\nLoss of: {np.abs(pred - y)}')
        cv.imshow("Input Image", img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
elif menu=="3":
    centeringErrors = []
    for i in range(testSize):
        x = np.expand_dims(xTestProcessed[i], axis=0)
        y = yTest[i]
        img = xTestProcessed[i]
        pred = model.predict(x)[0]
        print(f'Feature {i}\nModel predicted: {pred}. Actual label: {y}\nLoss of: {np.abs(pred - y)}')
        cv.imshow("Input Image", img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        centeringError = np.mean(pred[0] - y[0])
        centeringErrors.append(centeringError)
    avgCenteringError = np.mean(centeringErrors)
    print(f"Average Centering Error: {avgCenteringError}")

    centeringError_pub = rospy.Publisher("avgCenteringError", Float32, queue_size=10)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        centeringError_pub.publish(avgCenteringError)
        rate.sleep()

xTestProcessed = tf.convert_to_tensor(xTestProcessed)
yTest = tf.convert_to_tensor(yTest)
testPerf = model.evaluate(xTestProcessed, yTest, batch_size=1, verbose=0)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()

plt.show()
