import tensorflow as tf
from geometry_msgs.msg import Twist
import numpy as np
import cv2 as cv

class Drive:
    def dyn_rcfg_cb(config, level):
        global speed, yaw, drive
        yaw = config.yaw
        speed = config.speed
        drive = config.drive
        return config

    def __init__(self):
        self.model = tf.keras.model.load_model("../../../../reu-actor/home/actor_ws/src/deepsteer_pkg/scripts/model_with_preprocessing")
        self.speed = speed
        self.yaw = yaw

    def predict(self, image):
        vel_msg = Twist()
        self.process(image)
        prediction = self.model.predict(image)

        #add stuff from prediction model

        if drive:
            vel_msg.linear.x = self.speed
            vel_msg.angular.z = self.yaw

    def __process(self, image):
        image = image[0:][720:]
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.resize(image, (100,100))
        image = np.expand_dims(image, axis=0)
        return image



