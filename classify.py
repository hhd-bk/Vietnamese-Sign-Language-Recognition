import cv2
import copy
import numpy as np
import os.path
import imutils
import argparse

from keras.models import load_model
from collections import deque
from sub1_classify import get_hsv_value, skin_extract
from sub2_classify import put_text_onscreen

# Load model to predict
model = load_model('model.h5')

# Characters
character = ['nothing', 'a', 'b', 'c', 'd', 'đ', 'e', 'g', 'h', 'i', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't' , 'u', 'v', 'x', 'y', '^', 'w']
# ??
temp = 0
# Check if the skin has been set before
if os.path.exists('skin_value.npy') == False:
    get_hsv_value()
# Load the skin HSV value
lower, upper = np.load('skin_value.npy')

sequence = ''
sequence1 = ''

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32, help="max buffer size")
args = vars(ap.parse_args())

# initialize the list of tracked points, the frame counter, and the coordinate deltas
pts = deque(maxlen=args["buffer"])
dau = ''
# set d1 to d10 = 0 (count from 0) 0->9
d = np.zeros(10, dtype=np.uint8)

count1 = 1
counter = 0
temp = 0
temp1 = 0
(dX, dY) = (0, 0)
direction = ""
(dirX, dirY) = ("", "")

img_sequence = np.zeros((200, 1200, 3), np.uint8)
# Start the webcam
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    copy_frame = copy.copy(frame)
    cv2.rectangle(frame, (0, 0), (300, 300), (0, 255, 0), 1)

    hand_extract = copy_frame[0:300, 0:300]
    skin = skin_extract(hand_extract, lower, upper)
    hand_data = cv2.resize(skin, (100, 100))
    pred_data = np.reshape(hand_data, (-1, 100, 100, 3))
    model_out = model.predict(pred_data)[0]
    max_index = np.argmax(model_out)
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
        if (temp == 30) and (c1 != "^") and (c1 != "w") :
            sequence += c1
            sequence1 = c1
            temp = 0
            d = np.zeros(10, dtype=np.uint8)
        if (temp == 30) and (c1 == "^") and (sequence1 == "a") :
            c1 = "â"
            sequence = sequence[:-1]
            sequence += c1
            sequence1=c1
            temp = 0
            d = np.zeros(10, dtype=np.uint8)
        if (temp == 30) and (c1 == "^") and (sequence1 == "e") :
            c1 = "ê"
            sequence = sequence[:-1]
            sequence += c1
            sequence1=c1
            d = np.zeros(10, dtype=np.uint8)
            temp = 0
        if (temp == 30) and (c1 == "^") and (sequence1 == "o") :
            c1 = "ô"
            sequence = sequence[:-1]
            sequence += c1
            temp = 0
            sequence1=c1
            d = np.zeros(10, dtype=np.uint8)
        if (temp == 30) and (c1 == "w") and (sequence1 == "o") :
            c1 = "ơ"
            sequence = sequence[:-1]
            sequence += c1
            sequence1=c1
            temp = 0
            d = np.zeros(10, dtype=np.uint8)
        if (temp == 30) and (c1 == "w") and (sequence1 == "u") :
            c1 = "ư"
            sequence = sequence[:-1]
            sequence += c1
            sequence1=c1
            temp = 0
            d = np.zeros(10, dtype=np.uint8)
    if (c1=="d") and (temp>=10):
        count1=0
    if (c1=="d") and (temp==10):
        print("dấu is ready")
    if (c1!="d") and (temp>=15):
        count1=1
    if (sequence1 == "a") and (dau == "ă"):
        c1 = "ă"
        sequence = sequence[:-1]
        sequence += c1
        sequence1 = c1
        dau = "nothing"

    if (sequence1 == "a") and (dau == "sắc"):
        c1 = "á"
        sequence = sequence[:-1]
        sequence += c1
        dau = "nothing"

    if (sequence1 == "ă") and (dau == "sắc"):
        c1 = "ắ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "â") and (dau == "sắc"):
        c1 = "ấ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "e") and (dau == "sắc"):
        c1 = "é"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "ê") and (dau == "sắc"):
        c1 = "ế"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "o") and (dau == "sắc"):
        c1 = "ó"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "ô") and (dau == "sắc"):
        c1 = "ố"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "ơ") and (dau == "sắc"):
        c1 = "ớ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "u") and (dau == "sắc"):
        c1 = "ú"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "ư") and (dau == "sắc"):
        c1 = "ứ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "i") and (dau == "sắc"):
        c1 = "í"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "y") and (dau == "sắc"):
        c1 = "ý"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 != "a") and (sequence1 != "ă") and (sequence1 != "â") and (sequence1 != "i") and (sequence1 != "o") and (sequence1 != "ơ") and (sequence1 != "ô") and (sequence1 != "u") and (sequence1 != "ư") and (sequence1 != "y") and (dau == "sắc") :
        dau = "nothing"
    if (sequence1 == "a") and (dau == "huyền"):
        c1 = "à"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "ă") and (dau == "huyền"):
        c1 = "ằ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "â") and (dau == "huyền"):
        c1 = "ầ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "e") and (dau == "huyền"):
        c1 = "è"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "ê") and (dau == "huyền"):
        c1 = "ề"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "o") and (dau == "huyền"):
        c1 = "ò"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "ô") and (dau == "huyền"):
        c1 = "ồ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "ơ") and (dau == "huyền"):
        c1 = "ờ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "u") and (dau == "huyền"):
        c1 = "ù"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "ư") and (dau == "huyền"):
        c1 = "ừ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "i") and (dau == "huyền"):
        c1 = "ì"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "y") and (dau == "huyền"):
        c1 = "ỳ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"

    if (sequence1 != "a") and (sequence1 != "ă") and (sequence1 != "â") and (sequence1 != "i") and (sequence1 != "o") and (sequence1 != "ơ") and (sequence1 != "ô") and (sequence1 != "u") and (sequence1 != "ư") and (sequence1 != "y") and (dau == "huyền") :
        dau = "nothing"
    if (sequence1 == "a") and (dau == "hỏi"):
        c1 = "ả"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "ă") and (dau == "hỏi"):
        c1 = "ẳ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "â") and (dau == "hỏi"):
        c1 = "ẩ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "e") and (dau == "hỏi"):
        c1 = "ẻ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "ê") and (dau == "hỏi"):
        c1 = "ể"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "o") and (dau == "hỏi"):
        c1 = "ỏ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "ô") and (dau == "hỏi"):
        c1 = "ổ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "ơ") and (dau == "hỏi"):
        c1 = "ở"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "u") and (dau == "hỏi"):
        c1 = "ủ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "ư") and (dau == "hỏi"):
        c1 = "ử"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "i") and (dau == "hỏi"):
        c1 = "ỉ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "y") and (dau == "hỏi"):
        c1 = "ỷ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 != "a") and (sequence1 != "ă") and (sequence1 != "â") and (sequence1 != "i") and (sequence1 != "o") and (sequence1 != "ơ") and (sequence1 != "ô") and (sequence1 != "u") and (sequence1 != "ư") and (sequence1 != "y") and (dau == "hỏi") :
        dau = "nothing"
    if (sequence1 == "a") and (dau == "ngã"):
        c1 = "ã"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "ă") and (dau == "ngã"):
        c1 = "ẵ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "â") and (dau == "ngã"):
        c1 = "ẫ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "e") and (dau == "ngã"):
        c1 = "ẽ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "ê") and (dau == "ngã"):
        c1 = "ễ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "o") and (dau == "ngã"):
        c1 = "õ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "ô") and (dau == "ngã"):
        c1 = "ỗ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "ơ") and (dau == "ngã"):
        c1 = "ỡ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "u") and (dau == "ngã"):
        c1 = "ũ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "ư") and (dau == "ngã"):
        c1 = "ữ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "i") and (dau == "ngã"):
        c1 = "ĩ"
        sequence = sequence[:-1]
        sequence += c1

        dau = "nothing"
    if (sequence1 == "y") and (dau == "ngã"):
        c1 = "ỹ"
        sequence = sequence[:-1]
        sequence += c1
        dau = "nothing"
    if (sequence1 != "a") and (sequence1 != "ă") and (sequence1 != "â") and (sequence1 != "i") and (sequence1 != "o") and (sequence1 != "ơ") and (sequence1 != "ô") and (sequence1 != "u") and (sequence1 != "ư") and (sequence1 != "y") and (dau == "hỏi") :
        dau = "nothing"

    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    c = max(cnts, default ='', key=cv2.contourArea)
    # determine the most extreme points along the contour
    if c != '':
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        # draw the outline of the object, then draw each of the
        # extreme points, where the left-most is red, right-most
        # is green, top-most is blue, and bottom-most is teal
        cv2.drawContours(frame, [c], -1, (0, 255, 255), 3)
        cv2.circle(frame, extLeft, 8, (0, 0, 255), -1)
        cv2.circle(frame, extRight, 8, (0, 255, 0), -1)
        cv2.circle(frame, extTop, 8, (255, 0, 0), -1)
        cv2.circle(frame, extBot, 8, (255, 255, 0), -1)
        if len(cnts) > 0:
            pts.appendleft(extTop)
        for i in np.arange(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            if counter >= 10 and i == 10 and pts[i-10] is not None:
                dX = pts[i-10][0] - pts[i][0]
                dY = pts[i-10][1] - pts[i][1]
                if np.abs(dX) > 20:
                    dirX = "East" if np.sign(dX) == 1 else "West"
                if np.abs(dY) > 20:
                    dirY = "North" if np.sign(dY) == 1 else "South"
                if dirX != "" and dirY != "":
                    direction = "{}-{}".format(dirY, dirX)
                else:
                    direction = dirX if dirX != "" else dirY
                if np.abs(dX) < 20:
                    dirX = "" if np.sign(dX) == 1 else ""
                if np.abs(dY) < 20:
                    dirY = "" if np.sign(dY) == 1 else ""
                if dirX == "" and dirY == "":
                    direction = "{}-{}".format(dirY, dirX)
                else:
                    direction = dirX if dirX == "" else dirY
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        if count1 == 0 :
            if dirY == "North" and dirX == "East":
                d[1] += 1
            if dirY == "North" and dirX == "West":
                d[2] += 1
            if d[2] >= 4:
                if dirY == "South" and dirX == "West":
                    d[7] += 1
                if d[7] >= 4:
                    print("Dấu ă")
                    count1=1
                    dau = "ă"
                    d = np.zeros(10, dtype=np.uint8)
            if dirY == "South" and dirX == "West":
                d[0] += 1
            if d[0]>=10:
                print("Dấu sắc")
                dau = "sắc"
                count1=1
                d = np.zeros(10, dtype=np.uint8)
            if d[0]>=2:
                if dirY == "North" and dirX == "West":
                    d[8] += 1
                if d[8] >= 2:
                    if dirY == "South" and dirX == "West" :
                        d[9]+=1
                    if d[9]>=2:
                        print("Dấu ngã")
                        count1=1
                        dau = "ngã"
                        d = np.zeros(10, dtype=np.uint8)
            if d[0] >= 2:
                if dirY == "North" and dirX == "East":
                    d[3] += 1
                if d[3] >=1:
                    if dirY == "North":
                        d[4] += 1
                    if d[4] >= 2:
                        print("Dấu hỏi")
                        count1=1
                        dau = "hỏi"
                        d = np.zeros(10, dtype=np.uint8)
            if d[1]>=12:
                print("Dấu huyền")
                count1=1
                dau = "huyền"
                d = np.zeros(10, dtype=np.uint8)
    if (count1!=0):
        d = np.zeros(10, dtype=np.uint8)

    img_sequence = put_text_onscreen(img_sequence, '%s' %(sequence.upper()), 20, 20)
    cv2.imshow('sequence', img_sequence)
    cv2.imshow('Mask', hand_data)
    cv2.imshow('Webcam', frame)
    counter += 1
    k = cv2.waitKey(2) & 0xFF
    if k == 27:
        break
    if k == 113:
        sequence = ''

cap.release()
cv2.destroyAllWindows()
