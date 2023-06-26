import tensorflow as tf
import numpy as np
import cv2 as cv
import pandas as pd

# Load image paths and labels from the CSV file
data_df = pd.read_csv("/home/brendan/catkin_ws/src/deepsteer/twist.csv") #CHANGE BASED ON YOUR COMPUTER
imgPaths = data_df['image_path'].values
labels = data_df[['linear_x', 'angular_z']].values

# Define image resize parameters
imgHeight = 100
imgWidth = 100

# Preprocess the images
def preprocessData(imgs, imgHeight, imgWidth):
    processedImgs = []
    for path in imgs:
        img = cv.imread(path)
        img = cv.resize(img, (imgWidth, imgHeight))
        img = img.astype(np.float32) / 255.0  # Normalize the pixel values
        processedImgs.append(img)
    processedImgs = np.array(processedImgs)
    return processedImgs

# Load the saved model
model = tf.keras.models.load_model("trained_model")

# Preprocess the new images
xTestProcessed = preprocessData(imgPaths, imgHeight, imgWidth)

# Perform prediction on the new images
predictions = model.predict(xTestProcessed)

# Print the predicted values and actual values
x = 0
z = 0
for i in range(len(predictions)):
    pred = predictions[i]
    actual = labels[i]
    error = [pred[0]-actual[0], pred[1]-actual[1]]
    print(f"Image {i+1}: Predicted values: {pred}, Actual values: {actual}, Error: {error}")
    x += error[0]
    z += error[1]
print(f"Average x Error: {x/len(predictions)}, Average z Error: {z/len(predictions)}")

# You can further process or use the predictions and actual values as per your requirements
