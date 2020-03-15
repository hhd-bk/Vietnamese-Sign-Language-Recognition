from keras.models import load_model

import numpy as np
import cv2
import copy
import os

from sub_classify import skin_extract, get_hsv_value

# Take 60 frames as the input to feed the trained model
def write_video():
    # Check the skin value
    if os.path.exists('skin_value.npy') == False:
        get_hsv_value()
    # Press s to start write the video input data
    press_s = 0
    # Load the skin value in npy
    lower , upper = np.load('skin_value.npy')
    # Frame counter
    count_frames = 0
    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('in_put.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (100, 100))
    # Enable the webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Taking frame from webcam
        _, frame = cap.read()
        # Copy the frame to avoid the line of the rectangle
        copy_frame = copy.copy(frame)
        # Draw rectangle for extract the hand
        cv2.rectangle(frame, (40, 40), (340, 340), (0, 255, 0), 1)
        # Take the hand from copy hand (avoid the green line of the rectangle)
        hand_extract = copy_frame[40:340, 40:340]
        # Resize before extract (minizie the backgound color)
        hand_extract = cv2.resize(hand_extract, (100, 100))
        # Extract the skin color
        hand_data = skin_extract(hand_extract, lower, upper)
        # Wirte to video data
        if press_s == 1:
            out.write(hand_data)
            count_frames += 1
            print(count_frames)
        # Show hand extraction and frame
        cv2.imshow('Mask', hand_data)
        cv2.imshow('Webcam', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            # key s is pressed
            press_s = 1
        if k == 27 or count_frames == 60:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return

def convert_video_to_frames():
    # Save all the frames in an array
    frames = []
    # Counting the frame in the video
    counter_frame = 0
    # Set first value
    success = True
    cap = cv2.VideoCapture('in_put.avi')
    while success:
        # Reading each frame in the video
        success, frame = cap.read()
        # print(frame.shape)
        # Adding each frame to list frames
        counter_frame +=1
        frames.append(frame)
        if counter_frame == 60:
            break
    # Release the video for the next video
    cap.release()
    # Add all the frames into a lists
    # Convert to numpy array
    frames = np.array(frames)
    print(frames.shape)
    np.save('in_put.npy', frames)
    return

# Take 60 frames for predicting: 0-40 => model make 41 prediction
# Using the model to predict 20 frames in each sequence data then return result array
def predictor():
    # Load the video
    sequence_data = np.load('in_put.npy')
    # Loading the model
    model = load_model('model-3d-end.h5')
    predict_results = np.zeros(41, dtype=np.uint8)
    for frame in range(0, 41):
        # Take each 20 frames from the sequence
        predict_data = sequence_data[frame:(frame+20)]
        # Reshape the data for model prediction
        predict_data = np.reshape(predict_data, (-1, 20, 100, 100, 3))
        # Predicting using the model
        model_predict = model.predict(predict_data)[0]
        # Take the max index in the model prediction array
        max_index = np.argmax(model_predict)
        # Append to the predict results
        predict_results[frame] = max_index
    # Collect the most frequent number in the predict_results
    counters = np.bincount(predict_results)
    result = np.argmax(counters)
    return result

# Offline classify
def classify_3d():
    write_video()
    convert_video_to_frames()
    index = predictor()
    return index

classes = ['nothing', 'dấu ớ', 'dấu sắc', 'dấu huyền', 'dấu hỏi', 'dấu ngã', 'dấu nặng']
character = classify_3d()
print(classes[character])

