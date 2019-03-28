import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Lambda
from keras.layers import Cropping2D
from keras.layers import Conv2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l2
import pickle

# Constants
steering_adjustment = 0.27
epochs              = 15
batch_size          = 128
bright_lower        = 0.3

# What determines a turn, Can be done better by taking a mean or some other metric
# of the positive values and in turn for the negative values
turn_thresh         = 0.15
turn_angle_noise    = (-0.10, 0.10)

# Preparing data from driving_log.csv (Sample training data)
colnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data = pandas.read_csv('/opt/carnd_p3/data/driving_log.csv', skiprows=[0], names=colnames)

center = np.array(['/opt/carnd_p3/data/' + x.strip() for x in data.center.tolist()])
left = np.array(['/opt/carnd_p3/data/' + x.strip() for x in data.left.tolist()])
right = np.array(['/opt/carnd_p3/data/' + x.strip() for x in data.right.tolist()])
steering = np.array(data.steering.tolist())

input_shape = mpimg.imread(center[0]).shape

# Split train and valid
# Shuffle center and steering. Using 10% of center camera images and steering angles for validation.
center_train, X_valid, steering_train, y_valid = train_test_split(center, steering, test_size=0.10, shuffle=True, random_state=100)

center_train = np.array(center_train)
steering_train = np.array(steering_train)

# Filtering going straight, left and right turns
a_right = steering_train[steering_train > turn_thresh]
d_right = center_train[steering_train > turn_thresh]

a_left = steering_train[steering_train < -turn_thresh]
d_left = center_train[steering_train < -turn_thresh]

a_straight = steering_train[(steering_train < turn_thresh ) & (steering_train > -turn_thresh)]
d_straight = center_train[(steering_train < turn_thresh ) & (steering_train > -turn_thresh)]

# To account for the discrepancy in the dataset between the straight, right, left turns
# Since the number of images with steering straight is much higher than the rest,
# turns aren't able to contribute to the CNN weights effectively

# Filter angles less than -turn_thresh and add right camera images into driving left list, with a sharper turn angle
# Left camera images can be added to the left driving list by making a softer turn angle than the center image
# The opposite is done for driving right list
# Could be done better than converting to list but dont have to worry about appending dimensions
a_left = a_left.tolist() + (steering[np.where(steering < -turn_thresh)] - steering_adjustment).tolist()
d_left = d_left.tolist() + (right[np.where(steering < -turn_thresh)]).tolist()

a_right = a_right.tolist() + (steering[np.where(steering > turn_thresh)] + steering_adjustment).tolist()
d_right = d_right.tolist() + (left[np.where(steering > turn_thresh)]).tolist()

# Combine images from center, left and right cameras
X_train = d_straight.tolist() + d_left + d_right
y_train = np.float32(a_straight.tolist() + a_left + a_right)

# Augmentation and preprocessing
def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rand = np.random.uniform(bright_lower, 1.0)
    hsv[:, :, 2] = rand * hsv[:, :, 2]
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img


# Flip image about the vertical axis
def flip(image, angle):
    new_image = cv2.flip(image, 1)
    new_angle = angle * (-1)
    return new_image, new_angle


# Generators: inspired from: https://medium.com/@ensembledme/writing-custom-keras-generators-fe815d992c5a
# Training generator: Shuffle training data before choosing data
# Go through each training sample in outer loop and create batch and return in inner loop
# This works better than using a completely Stocastic method
# Apply random brightness into the chosen sample, Add some small random noise for the chosen angle
def generator_data(batch_size):
    batch_train = np.zeros((batch_size, *input_shape), dtype=np.float32)
    batch_angle = np.zeros((batch_size,), dtype=np.float32)
    while True:
        train_data, angle = shuffle(X_train, y_train)
        for i in range(len(X_train) // batch_size):
            for j in range(batch_size):
                choice = int(np.random.choice(len(train_data), 1))
                # Adding random brightness changes to the training image
                batch_train[j] = random_brightness(mpimg.imread(train_data[choice].strip()))
                # Adding noise to the steer angle
                batch_angle[j] = angle[choice] * (1 + np.random.uniform(*turn_angle_noise))
                # Flip random images
                if np.random.randint(2) == 1:
                    batch_train[j], batch_angle[j] = flip(batch_train[j], batch_angle[j])
            yield batch_train, batch_angle


# Validation generator: Pick random samples and create a batch
def generator_valid(valid_data, angle, batch_size):
    batch_train = np.zeros((batch_size, *input_shape), dtype=np.float32)
    batch_angle = np.zeros((batch_size,), dtype=np.float32)
    while True:
        data, angle = shuffle(valid_data, angle)
        for i in range(batch_size):
            rand = int(np.random.choice(len(valid_data), 1))
            batch_train[i] = mpimg.imread(valid_data[rand].strip())
            batch_angle[i] = angle[rand]
        yield batch_train, batch_angle


def nvidia_model():
    # Training Architecture: inspired by the NVIDIA End to End model
    model = Sequential()
    model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=input_shape))
    model.add(Lambda(lambda x: x / 255 - 0.5))
    model.add(Conv2D(24, (5, 5), strides = (2, 2), activation='relu', kernel_regularizer = l2(0.001)))
    model.add(Conv2D(36, (5, 5), strides = (2, 2), activation='relu', kernel_regularizer = l2(0.001)))
    model.add(Conv2D(48, (5, 5), strides = (2, 2), activation='relu', kernel_regularizer = l2(0.001)))
    model.add(Conv2D(64, (3, 3), strides = (2, 2), activation='relu', padding='same',  kernel_regularizer = l2(0.001)))
    model.add(Conv2D(64, (3, 3), strides = (2, 2), activation='relu', kernel_regularizer = l2(0.001)))
    model.add(Flatten())
    model.add(Dense(100, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(50, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(16, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(10, kernel_regularizer=l2(0.001)))
    model.add(Dense(1, kernel_regularizer=l2(0.001)))
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    model.summary()
    return model


# Training
def train():
    data_generator = generator_data(batch_size)
    valid_generator = generator_valid(X_valid, y_valid, batch_size)

    model = nvidia_model()
    
    filepath="model-check.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True)
    callbacks_list = [checkpoint]

    history_fit = model.fit_generator(data_generator, steps_per_epoch = len(X_train) // batch_size,
                        epochs=epochs, validation_data=valid_generator, validation_steps=len(X_valid),
                        callbacks=callbacks_list)
    
    print('Saving model ...')
    
    model.save("model.h5")
    
    print('Model saved!')
    
    # Used later to plot the validation vs training loss
    with open('hist.p', 'wb') as file_pi:
        pickle.dump(history_fit.history, file_pi)

if __name__ == '__main__':
    train()