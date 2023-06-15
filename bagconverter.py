import tensorflow as tf
import numpy as np
import os
import cv2
import rosbag
from sensor_msgs.msg import Image
import shutil
from cv_bridge import CvBridge
import tensorflow as tf
from pathlib import Path
import datetime

DATASET_DUMP_DIR = "../rosbags"
IMAGE_TOPIC = "/camera/image_raw"
VEH_STEERING_TOPIC = "/vehicle/steering_report"
IMAGE_DUMP_DIR = "img/training/"
STEER_IMAGE_RATIO = 10
TRAIN_SPLIT = 0.9
DATE_AND_TIME = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def extract_images(bag_file):
    
    # extract_images will open your bag file, then label and dump all of your images
    # parameters None
    # returns None

    print("extract_images invoked...")

    # Instantiate bag and bridge
    bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()

    steering_angles = []
    velocity = []
    mv = []
    msa = []
    it = 0

    # Extract steering_report
    print("Extracting steering angles...")

    for topic, msg, t in bag.read_messages(topics=[VEH_STEERING_TOPIC]):
        msa.append(float(msg.steering_report.steering_wheel_angle))
        mv.append(float(msg.speed))

        if it % STEER_IMAGE_RATIO == 0:
            # Take the average of the messages around this frame
            mean_angle = sum(msa) / STEER_IMAGE_RATIO
            mean_velocity = sum(mv) / STEER_IMAGE_RATIO
            msa = []
            mv = []
            
            steering_angles.append(str(float(mean_angle)))
            velocity.append(str(float(mean_velocity)))

        it += 1

    it = 0

    # Extract images
    print("Extracting images...")
    for topic, msg, t in bag.read_messages(topics=[IMAGE_TOPIC]):
        
        fn = os.path.join(IMAGE_DUMP_DIR, steering_angles[it] + "_" + velocity[it] + "_" + str(DATE_AND_TIME) + ".png")

        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        cv2.imwrite(fn, cv_img)

        it += 1

    bag.close()


def get_image(file_path):
    
    file_name = file_path.split('/')[-1]

    velocity = float(file_name.split('_')[1])
    velocity = np.array([velocity])
    velocity = np.expand_dims(velocity, axis=0)


    img = cv2.imread(file_path)
    img = img[0:][720:][0:]
    img = cv2.resize(img, (100,100))
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)
    img /= 255

    return img, velocity

def get_label(file_path):
    file_name = file_path.split('/')[-1]
    label = float(file_name.split('_')[0])
    label = np.array([label], dtype="float32")

    return label

def dataset_gen():
    all_images = list(Path(IMAGE_DUMP_DIR).glob('*.png'))

    for img_path in all_images:
        img_path = str(img_path)
        img, velocity = get_image(img_path)
        label = get_label(img_path)
        yield img, velocity, label

def standardize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    standardized = (data - mean) / std
    return standardized

def standard_gen():
    global images, velocities, labels
    for i, e in enumerate(images):
        yield (e, velocities[i]), labels[i]


def get_unique_dir(train = "train"):
    subdirs = sorted(os.listdir(DATASET_DUMP_DIR))
    subdirs = [x for x in subdirs if train in x]

    if len(subdirs) > 0:
        last_dataset_number = subdirs[-1][-4:]
        dir_number = str(int(last_dataset_number) + 1)
        for a in range(4 - len(dir_number)):
            dir_number = "0" + dir_number
    else:
        dir_number = "0000"
    
    dir_name = os.path.join(DATASET_DUMP_DIR, "tfds_" + train + dir_number)

    return str(dir_name)

if __name__ == "__main__":
    all_bags= None
    choose_bag = "y"
    while choose_bag == "y":
        choose_bag = input("Extract ros bag? (y/n/*): ")
        if choose_bag == "y":
            all_bags = list(Path(".").glob("*.bag")) if all_bags == None else all_bags
            for (i, e) in enumerate(all_bags):
                print(f"{i+1}: {e}")
            bag_file = all_bags[int(input("Choose a bag (enter a number): "))]
            all_bags = [x for x in all_bags if x != bag_file]
            extract_images(bag_file)
        elif choose_bag == "*":
            all_bags = list(Path(".").glob("*.bag")) if all_bags == None else all_bags
            for i in all_bags:
                extract_images(i)

    print("Generating dataset (non-standard)...")

    ds = tf.data.Dataset.from_generator(dataset_gen, (tf.float32, tf.float32, tf.float32))

    '''print()
    for img, velocity, label in ds.take(5):
        print(f"img shape: {img.shape}")
        print(f"velocity: {velocity[0]}")
        print(f"label: {label[0]}")
    print()'''

    print("Extracting tensors...")

    images = [images for images, velocities, labels in ds]
    velocities = [velocities for images, velocities, labels in ds]
    labels = [labels for images, velocities, labels in ds]

    print("Standardizing data...")

    velocities = standardize_data(velocities)
    labels = standardize_data(labels)

    print("Generating dataset (standardized)")

    ds = tf.data.Dataset.from_generator(standard_gen, ((tf.float32, tf.float32), tf.float32))

    ds = ds.shuffle(buffer_size=50)
    ds_size = sum(1 for _ in ds)
    print("Cardinality:", ds_size)
    ds_train_features = int(ds_size * TRAIN_SPLIT)

    for (img, vel), lab in ds.take(1):
        input(f'Img shape: {img.shape}\nVel shape: {vel.shape}\nLabel shape: {lab.shape}')

    # Split data into train and test
    ds_train = ds.take(ds_train_features)
    ds_test = ds.skip(ds_train_features)

    ds_train = ds_train.prefetch(buffer_size = tf.data.AUTOTUNE)
    ds_test = ds_test.prefetch(buffer_size = tf.data.AUTOTUNE)

    # Save datasets with a unique directory
    directory_name = get_unique_dir("train")
    print(f"Saving dataset as {directory_name} with {ds_train_features} training features")
    ds_train.save(directory_name)
    directory_name = get_unique_dir("test")
    print(f"Saving dataset as {directory_name} with {ds_size-ds_train_features} testing features")
    ds_test.save(directory_name)