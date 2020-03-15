import numpy as np
import os
import cv2

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img

from random import shuffle
# Convert all jpg files into npy file: X_data.npy
# Convert one-hot label into npy file: Y_data.npy
# The components in X correspond to the components in Y in order
def convert_image_data():
    # path_dir      : The directory stores jpg files
    path_dir = 'Data\\2d\\Test'
    # num_classes   : The number of classes
    num_classes = 26
    # img_size      : The resolution of image after resized
    img_size = 100
    # start         : The first jpg file to convert
    start = 0
    # end           : The last jpg file to convert
    end = 1000
    # X_data        : Name of npy file store images
    X_data = 'X_test.npy'
    # Y_data        : Name of npy file store labels
    Y_data = 'Y_test.npy'

    # List for storing image
    img_data = []
    # List for storing labels
    labels = []
    # Array from 0 to num_classes-1 coresspond to each class
    classes = np.arange(0, num_classes)

    for i in classes:
        # Take the path of each class directory data
        cur_dir = os.path.join(path_dir, '%d' %i)
        # List of jpg file in the cur_dir
        path_list = os.listdir(cur_dir)
        for img in path_list[start:end]:
            # Take the path of each image
            path = os.path.join(cur_dir, img)
            # Reading each image
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            # Resize into image img_size*img_size
            image = cv2.resize(image, (img_size, img_size))

            # Store each image in order of classes into list img_data
            img_data.extend([image])
            # Store label of each image respective with img_data
            labels.extend([i])

    # Convert into numpy array
    X = np.array(img_data)
    Y = np.array(labels)

    print(X.shape)
    print(Y.shape)

    # Saving data into npy format
    np.save('%s' %X_data, X)
    np.save('%s' %Y_data, Y)
    return

# Visualizing augmented data
# Augment data from train_dir then save it in save_dir
# Each jpg file in train_dir is mutiplied with count number
def aug_image(train_dir, save_dir, count):
    # train_dir : The directory store raw data taken from webcam
    train_dir = 'Data\\2d\\Raw'
    # save_dir  : The directory for saving augmented data
    save_dir = 'Data\\2d\\Aug'
    # count     : The mutiply of each image
    count = 5

    # Random rotation range in degree
    rotation = 20
    # Random shift in width and height (value can be in range from 0 to 1)
    shift_width = 0.25
    shift_height = 0.25

    # Augment image functoin in keras library
    datagen = ImageDataGenerator(
                                 rotation_range = rotation,             # Degree range for random rotations
                                 width_shift_range = shift_width,       # Shift in width
                                 height_shift_range = shift_height,     # Shift in height
                                 )

    # List of image in train_dir
    image_path_list = []
    for file in os.listdir(train_dir):
        image_path_list.append(os.path.join(train_dir, file))

    # Save data into save_dir
    k = 0 # Counter for new image (augmented image)
    for img in image_path_list:
        # Loads an image into numy array
        image = img_to_array(load_img(img))
        # Reshape the image for feeding into data flow (1, img_size, img_size, 3) 1 mean 1 image
        image = image.reshape((1,) + image.shape)
        datagen.fit(image)
        # Feed image for augmenting, batch_size = 1 mean 1 image each time
        images_flow = datagen.flow(image, batch_size=1)
        # save new images for visualizing
        for i, new_images in enumerate(images_flow):
            output_path = os.path.join(save_dir, "AUG{}.png")
            k = k + 1
            i = i + 1
            new_image = array_to_img(new_images[0], scale=True)
            new_image.save(output_path.format(k))
            if i >= count:
                break
    return

# Convert data to npy file to feed the neural network
# Take each frame from video then put it into an array
def convert_video_data():
    # Save all the video in the training data
    vid_frames = []
    # Save the labels
    labels = []
    # Reading the video data from directory
    for classes in range(0, 7):
        # Reading each video in each class
        for vid in os.listdir('Data\\3d\\Raw\\%d' %classes):
            # Store each frame in the video
            frames = []
            # Frame counting
            counter_frame = 0
            # Video path
            vid_path = os.path.join('Data\\3d\\Raw\\%d' %classes, vid)
            # Open the video
            cap = cv2.VideoCapture(vid_path)
            # Each video has 20 frames 0 -> 19
            while True:
                # Reading each frame in the video
                success, frame = cap.read()
                # print(frame.shape)
                # Adding each frame to list frames
                frames.extend([frame])
                counter_frame += 1
                if counter_frame == 20:
                    break
            # Release the video for the next video
            cap.release()
            # Add all the frames into a lists
            vid_frames.extend([frames])
            # The labels for each class
            labels.extend([classes])
            # Convert to numpy array
            np.save('X_train_3d.npy', np.array(vid_frames))
            np.save('Y_train_3d.npy', np.array(labels))
    return
convert_image_data()
