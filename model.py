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
from keras.optimizers import Adam
from keras.regularizers import l2

# Constants
steering_adjustment = 0.27
epochs = 3
batch_size = 128
# What determines a turn, Can be done better by taking a mean of the positive values and in turn for the negative values
turn_thresh = 0.15

# Preparing data from driving_log.csv (Sample training data)
colnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data = pandas.read_csv('/opt/carnd_p3/data/driving_log.csv', skiprows=[0], names=colnames)

center = np.array(['/opt/carnd_p3/data/' + x.strip() for x in data.center.tolist()])
left = np.array(['/opt/carnd_p3/data/' + x.strip() for x in data.left.tolist()])
right = np.array(['/opt/carnd_p3/data/' + x.strip() for x in data.right.tolist()])
steering = np.array(data.steering.tolist())

input_shape = mpimg.imread(center[0])[60:140,:].shape
# input_shape = (64, 64, 3)

# Split train and valid
# Shuffle center and steering. Using 10% of central images and steering angles for validation.
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
a_left.tolist().extend((steering[np.where(steering < -turn_thresh)] - steering_adjustment).tolist())
d_left.tolist().extend((right[np.where(steering < -turn_thresh)]).tolist())

a_right.tolist().extend((steering[np.where(steering > turn_thresh)] + steering_adjustment).tolist())
d_right.tolist().extend((left[np.where(steering > turn_thresh)]).tolist())

# Combine images from center, left and right cameras
X_train = d_straight.tolist() + d_left.tolist() + d_right.tolist()
y_train = np.float32(a_straight.tolist() + a_left.tolist() + a_right.tolist())


# Augmentation and preprocessing
def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rand = np.random.uniform(0.3, 1.0)
    hsv[:, :, 2] = rand * hsv[:, :, 2]
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img


# Flip image about the vertical axis
def flip(image, angle):
    new_image = cv2.flip(image, 1)
    new_angle = angle * (-1)
    return new_image, new_angle


def crop_resize(image):
  return image[60:140,:]


# Generators: inspired from: https://medium.com/@ensembledme/writing-custom-keras-generators-fe815d992c5a
# Training generator: Shuffle training data before choosing data,
# pick random training data to feed into batch at each "for" loop.
# Apply random brightness, resize, crop into the chosen sample. Add some small random noise for chosen angle.
def generator_data(batch_size):
    batch_train = np.zeros((batch_size, *input_shape), dtype=np.float32)
    batch_angle = np.zeros((batch_size,), dtype=np.float32)
    while True:
        data, angle = shuffle(X_train, y_train)
        for i in range(batch_size):
            choice = int(np.random.choice(len(data), 1))
            # Adding random brightness changes to the training image
            batch_train[i] = crop_resize(random_brightness(mpimg.imread(data[choice].strip())))
            # Adding noise to the steer angle
            batch_angle[i] = angle[choice] * (1 + np.random.uniform(-0.05, 0.05))
            # Flip random images
            if np.random.randint(2) == 1:
                batch_train[i], batch_angle[i] = flip(batch_train[i], batch_angle[i])
        yield batch_train, batch_angle


# Validation generator: pick random samples. Apply resizing and cropping on chosen samples
def generator_valid(data, angle, batch_size):
    batch_train = np.zeros((batch_size, *input_shape), dtype=np.float32)
    batch_angle = np.zeros((batch_size,), dtype=np.float32)
    while True:
        data, angle = shuffle(data, angle)
        for i in range(batch_size):
            rand = int(np.random.choice(len(data), 1))
            batch_train[i] = crop_resize(mpimg.imread(data[rand].strip()))
            batch_angle[i] = angle[rand]
        yield batch_train, batch_angle


def nvidia_model():
    # Training Architecture: inspired by the NVIDIA End to End model
#     input_size = 64
    model = Sequential()
    
    # Doesn't get serialized and had to run the network again, included it in a different method
#     model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=input_shape))
#     model.add(Lambda(lambda image: tf.image.resize_images(image, (input_size, input_size))))

    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=input_shape))
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
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    model.summary()
    return model


# Training
def train():
    data_generator = generator_data(batch_size)
    valid_generator = generator_valid(X_valid, y_valid, batch_size)

    model = nvidia_model()

    model.fit_generator(data_generator, steps_per_epoch=len(X_train) // batch_size,
                        epochs=epochs, validation_data=valid_generator, validation_steps=len(X_valid))

    print('Done Training')

    model.save("model.h5")

if __name__ == '__main__':
    train()