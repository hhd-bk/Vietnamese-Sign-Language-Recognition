import cv2
import copy
import numpy as np
import os.path
from keras.models import load_model
from sub1_classify import skin_extract, get_hsv_value
from sub2_classify import put_text_onscreen

# Real time classify
def main_program():
    # Table of character and class order
    character = ['nothing', 'a', 'b', 'c', 'd', 'đ', 'e', 'g', 'h', 'i', 'k', 'l', 'm',
    #             0          1    2    3    4    5    6    7    8    9    10   11   12
                 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', '^', 'w']
    #             13   14   15   16   17   18   19    20   21   22  23   24   25
    if os.path.exists('skin_value.npy') == False:
        get_hsv_value()
    # Load the skin value in HSV coordinate
    lower, upper = np.load('skin_value.npy')

    # Load the model to predict
    model = load_model('Best.h5')
    # 11
    img_sequence = np.zeros((200, 1200, 3), dtype=np.uint8)
    temp = 0
    sequence = ''
    sequence1 = ''
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
        # Reshape for feeding the model to predict
        pred_data = np.reshape(hand_data, (-1, 100, 100, 3))
        # The proability of all classes
        model_out = model.predict(pred_data)[0]
        # Find the index have max proability
        max_index = np.argmax(model_out)
        c1 = ''

        frame = put_text_onscreen(frame, '%s' % character[max_index], 400, 400)
        c1 = character[max_index]
        if (c1 != "nothing" ):
            if (temp == 0):
                previousc1 = c1
            if previousc1 == c1:
                previousc1 = c1
                temp += 1
            else:
                temp = 0
            if (temp == 20) and (c1 != "^") and (c1 != "w") :
                sequence += c1
                sequence1 = c1
                temp = 0
                print (sequence)
            if (temp == 20) and (c1 == "^") and (sequence1 == "a") :
                c1 = "â"
                sequence = sequence[:-1]
                sequence += c1
                temp = 0
                print (sequence)
            if (temp == 20) and (c1 == "^") and (sequence1 == "e") :
                c1 = "ê"
                sequence = sequence[:-1]
                sequence += c1
                temp = 0
                print (sequence)
            if (temp == 20) and (c1 == "^") and (sequence1 == "o") :
                c1 = 'ô'
                sequence = sequence[:-1]
                sequence += c1
                temp = 0
                print (sequence)
            if (temp == 20) and (c1 == "w") and (sequence1 == "o") :
                c1 = "ơ"
                sequence = sequence[:-1]
                sequence += c1
                temp = 0
                print (sequence)
            if (temp == 20) and (c1 == "w") and (sequence1 == "u") :
                c1 = "ư"
                sequence = sequence[:-1]
                sequence += c1
                temp = 0
                print (sequence)

        img_sequence = put_text_onscreen(img_sequence, '%s' %(sequence.upper()), 20, 20)
        cv2.imshow('sequence', img_sequence)
        # Show mask and hand data
        cv2.imshow('Mask', hand_data)
        cv2.imshow('Webcam', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == 113:
            sequence = ''
    cap.release()
    cv2.destroyAllWindows()
    return

# Run the main program
main_program()
