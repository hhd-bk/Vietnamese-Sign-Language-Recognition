import numpy as np
from keras.models import load_model
import os

# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# model = load_model('Model\\2d\\Full.h5')
# model.summary()

def off_classify():
    model = load_model('Best.h5')
    test_data = np.load('X_test.npy')
    label_test_data = np.load('Y_test.npy')
    counter = 0
    index = 0
    for classes in range(0, 26):
        for data in test_data[0+classes*1000:1000+classes*1000]:

            pred_data = np.reshape(data, (-1, 100, 100, 3))
            model_out = model.predict(pred_data)[0]
            max_index = np.argmax(model_out)
            if  max_index == label_test_data[index]:
                counter += 1
            index += 1
        print('class %d: %d' %(classes, counter))
        counter = 0
    return

off_classify()
