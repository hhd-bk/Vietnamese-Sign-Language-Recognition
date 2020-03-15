import cv2
import copy
import numpy as np
from sub_classify import skin_extract

# Enable the webcam
cap = cv2.VideoCapture(0)

# Table of character and class order
character = ['nothing', 'a', 'b', 'c', 'd', 'dd' , 'e', 'g', 'h', 'i', 'k', 'l', 'm',
#             0          1    2    3    4    5      6    7    8    9    10   11   12
             'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', '^', 'w']
#             13   14   15   16   17   18   19   20   21   22   23   24   25

# Load hsv value for skin extractiom
lower, upper = np.load('skin_value.npy')

# Swtich for saving data (press s)
s = 0
# Swtich for creting next class (press c)
ch_cl = 0

# end - pic_num = the number of images save each time press s
pic_num = 1
end = 1001

# Image resolution
img_size = 100
# First class to create data
class_create = 20

while True:
    _, frame = cap.read()
    # Copy the frame
    copy_frame = copy.copy(frame)
    # Draw rectangle for extract the hand
    cv2.rectangle(frame, (40, 40), (340, 340), (0, 255, 0), 1)
    # Take the hand from copy hand (avoid the green line of the rectangle)
    hand = copy_frame[40:340, 40:340]
    # Take the hand skin)
    hand = cv2.resize(hand, (img_size, img_size))
    skin_data = skin_extract(hand, lower, upper)


    # Taking the frame
    if s == 1:
        # cv2.imwrite('Data\\2d\\Valid\\%d\\%d.png' %(class_create, pic_num), skin_data, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite('Data\\2d\\Test\\%d\\%d.jpg' % (class_create, pic_num), skin_data,[int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print('Saved %d' %pic_num)
        pic_num += 1
        if pic_num == end:
            s = 0
            end = end + 1
    if ch_cl == 1 or pic_num == 1000:
        class_create += 1
        pic_num = 1
        end = 1001
        ch_cl = 0
        print('Class creating: %d' %class_create)
    cv2.imshow('Mask', skin_data)
    cv2.imshow('Webcam', frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    if k == ord('s'):
        s = 1
    if k == ord('c'):
        ch_cl = 1
cap.release()
cv2.destroyAllWindows()
