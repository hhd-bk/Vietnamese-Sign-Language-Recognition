import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten, Conv2D, AveragePooling2D
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam
from sklearn.utils import shuffle

# Define the model
def cnn_model():
    # Model graph

    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(100, 100, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))
    # Softmax for many classes
    model.add(Dense(26, activation='softmax'))
    return model

# Load image and labels
def load_data(X, Y):
    X = np.load('%s.npy' %X)
    Y = np.load('%s.npy' %Y)
    # Convert to one hot array label
    Y = np_utils.to_categorical(Y, 26)
    # Shuffle data
    X, Y = shuffle(X, Y)
    return X, Y

# Generator to yeild batch size of image to train
def train_flow(batch):
    # Load images to be augmented
    X_aug, Y_aug = load_data('X_aug', 'Y_aug')
    # Shuffle the training data set
    X_raw, Y_raw = load_data('X_raw', 'Y_raw')

    # Parameters for augmention
    aug_data = ImageDataGenerator(
        rotation_range=35,          # Random rotation range in degree
        width_shift_range=0.3,     # Random shift in width and height (value can be in range from 0 to 1)
        height_shift_range=0.3,
        zoom_range=0.15)             # Zoom range (value can be from 0 to 1)
    # Using raw data no parameter for augmention
    raw_data = ImageDataGenerator()

    # batch size of augmented, raw1 and raw2 data
    batch_size_aug = int(batch * 0.5)
    batch_size_raw = batch - batch_size_aug

    # Infinity loop in generating batch_size*augment_percent of augmented data
    aug_flow = aug_data.flow(X_aug, Y_aug, batch_size=batch_size_aug)
    raw_flow = raw_data.flow(X_raw, Y_raw, batch_size=batch_size_raw)

    while True:
        # Get the next batch size of augmented and raw data
        X_aug, Y_aug = aug_flow.next()
        X_raw, Y_raw = raw_flow.next()

        # Add two small batch size of data together
        X_train_next = np.concatenate([X_aug, X_raw], axis=0)
        Y_train_next = np.concatenate([Y_aug, Y_raw], axis=0)

        # Each time execute return a batch_size of data
        yield X_train_next, Y_train_next

# Plot and save the loss and acc graph during training process
def plot_hist(hist):
    plt.figure(1)
    plt.plot(hist.history['loss'])
    plt.title('Cross entropy training loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['training set'], loc='upper right')
    plt.savefig('training_loss_entropy_graph.png', bbox_inches='tight')

    plt.figure(2)
    plt.plot(hist.history['val_loss'])
    plt.title('Cross entropy validation loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['validation set'], loc='upper right')
    plt.savefig('validation_loss_entropy_graph.png', bbox_inches='tight')

    plt.figure(3)
    plt.plot(hist.history['acc'])
    plt.title('Training accuracy')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['training set'], loc='upper right')
    plt.savefig('training_accuracy_graph.png', bbox_inches='tight')

    plt.figure(4)
    plt.plot(hist.history['val_acc'])
    plt.title('Validation accuracy')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['validation set'], loc='upper right')
    plt.savefig('validation_accuracy_graph.png', bbox_inches='tight')
    return

# Train the model
def train_model(nb_epochs, lr, decay, batch, re_train):
    # Load validation data set
    X_valid, Y_valid = load_data('X_valid', 'Y_valid')

    # Create a geneator for training data
    train_data_flow = train_flow(batch)

    if re_train == True:
        model = load_model('model-2d.h5')
    else:
        # Get the model aritechture
        model = cnn_model()

    # Setup for the training process
    adam_op = adam(lr=lr, decay=decay)  # Advanced gradient descent
    model.compile(loss='categorical_crossentropy', optimizer=adam_op, metrics=['accuracy'])

    # Define callbacks: check point to save best model and stop when model is fitted
    check_points = ModelCheckpoint('model-2d.h5', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto')

    # Train model
    hist = model.fit_generator(train_data_flow,
                               steps_per_epoch=25,             # (240+500)*26 =19240; +1 for loop all over the data
                               epochs=nb_epochs,
                               validation_data=(X_valid, Y_valid),
                               callbacks=[check_points, early_stopping],     # save the best model base on validation data
                               verbose=1)

    plot_hist(hist)
    return

# Number of epochs
epochs = 50
# Learning rate (Decrease when loss being constant or loss not decrese)
learning_rate = 0.0005
# Number of images feed into model each iteration (Increse this value to reduce 'noise')
batch_size = 512
# Decay rate
decay_rate = 0.01
# Training the model
train_model(epochs, learning_rate, decay_rate, batch_size, re_train=False)


