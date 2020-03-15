from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def cnn_model():
    model = Sequential()

    model.add(Conv3D(16, (3, 3, 3), input_shape=(20, 100, 100, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(7, activation='softmax'))
    model.summary()

    return model

def train_model(batch, lr, nb_epochs):
    # Loading the training data
    X_train = np.load('X_train_3d.npy')
    Y_train = np.load('Y_train_3d.npy')

    print(X_train.shape)
    print(Y_train.shape)

    X_train, Y_train = shuffle(X_train, Y_train)
    Y_train = np_utils.to_categorical(Y_train, 7)

    # Loading the validation data
    # X_valid = np.load('X_valid_3d.npy')
    # Y_valid = np.load('Y_valid_3d.npy')

    # Load the model
    model = cnn_model()

    # The opimizier
    adam_op = adam(lr=lr)
    # Compile the model:
    model.compile(loss='categorical_crossentropy', optimizer=adam_op, metrics=['accuracy'])

    # Create a checkpointer for saving model with best performance in validation data set
    checkpointers = ModelCheckpoint('model-3d.h5', verbose=1, save_best_only=True, mode='auto')

    # Training the model and saving training process
    hist = model.fit(X_train, Y_train,
                     batch_size=batch,
                     epochs=nb_epochs,
                     verbose=1,
    #                  validation_data=(X_valid, Y_valid),
                     callbacks=[checkpointers])

    model.save('model-3d-end.h5')

    # Plot and save the loss graph during training process
    plt.figure(1)
    plt.plot(hist.history['loss'])
    plt.title('Cross entropy training loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend('training set', loc='upper right')
    plt.savefig('training_loss_entropy_graph.png', bbox_inches='tight')

    plt.figure(2)
    plt.plot(hist.history['val_loss'])
    plt.title('Cross entropy validation loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend('validation set', loc='upper right')
    plt.savefig('validation_loss_entropy_graph.png', bbox_inches='tight')
    return

# Batch_size
batch_size = 8
# Learning rate
learning_rate = 0.00001
# Epochs
epochs = 50

train_model(batch_size,learning_rate, epochs)