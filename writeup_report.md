# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior / Use sample data and augment it to obtain enough data
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---
[//]: # (Image References)
[image1]: ./writeup_images/data_dist_prev.png "Original distribution of data"
[image2]: ./writeup_images/data_dist_post.png "Final distribution of data"
[image3]: ./writeup_images/train_val_loss.png "Training vs Validation loss"



### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 demonstrating the car driving autonomously on the track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 104-108) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 103). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 111, 113 etc). 

The model was trained and validated using data set generators that added random brightness and other augmentations and noise to diversify the data and ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer. After some experimentation the learning rate of 0.0001 proved to be the best. (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from oversteer. For recovery I added left camera data to the right turn data by adding a recovery angle to the steering label, and vice versa to the left turn data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was inspired by the NVIDIA end to end learning model
I thought this model might be appropriate because it was a tried and tested model for this exact purpose and was tested in the wild on real cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

#### Regularization:
To combat the overfitting, the model in itself has a variety of regularization techniques such as dropout, weight regularizers like the l2 regularizer.

Then I modified a few layers to reduce the size of the model since the size of the dataset was pretty small in this case as compared to the original model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I modified the learning rate, and made sure the generator was using all the training data. (Initially I was using a completely stocastic generator and that might have caused it to not use the entire dataset.)

I also discovered a bug much later that wasn't appending the recoery images. Once that was fixed, post-training, the model ran just fine.

The a very small learning rate turned out to be the best choice in this case. But since many augmentation / regularization techniques were used, the dataset turned out to be comprehensive and the model was fully trained in 15 epochs.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 103-120) consisted of a convolution neural network with the following layers and layer sizes:
```sh
________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 80, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 38, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 17, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 19, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 9, 64)          36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 576)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               57700     
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_2 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 16)                816       
_________________________________________________________________
dropout_3 (Dropout)          (None, 16)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 10)                170       
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11        
=================================================================
Total params: 195,095
Trainable params: 195,095
Non-trainable params: 0
```
#### 3. Creation of the Training Set & Training Process

I have used only the sample data provided to us in the sample data folder of the class but a variety of augmentation and recovery techniques were used so that the neural network generalizes well.

I randomly shuffled the data set and put 10% of the data into a validation set. 

I then separated the training set images into driving straight, left and right first using a threshold of 0.15 magnitude for right and left turns. This I can later use to augment left and right camera images to the right and left turn partitions respectively for recovery steps.

The following is the visualization of the distribution of the original sample dataset. As we can see there is clear discrepancy that would cause the vehicle to be biased towards driving straight for the most part. I have used a variety of techniques to mitigate this limitation but at the same time the vehicle will be mostly driving straight on the track so this augmentation should be done carefully.

![alt text][image1]

After adding the recovery images the following is the distribution. In addition to the below, random brightness, flipping the images add to the dataset to make it more diverse.

![alt text][image2]


#### Recovery:
I took the idea in the concept 13 of the Behavioral cloning section and experimented with different values of the steering adjustment angle. Here I have only augmented the turning right data with the left camera images and increasing the intensity of the turn by the correcting factor and vice versa but I could have also added the left camera images to the driving left list by reducing the intensity of the turn.

After the collection process, I had 8727 number of data points in the training set. 

#### Preprocessing:
I then preprocessed this data by cropping the top 60 rows which mostly had the sky and trees and the bottom 20 rows of the images which just had the bonnet of the car. This proved to be a very important step in the preprocessing.

#### Normalization:
I also have a normalization layer at the beginning of the neural network which keeps the intensity values of the pixels between 0 and 1. This helps in faster training.

#### Data Augmentation:
In order to augment the images further, I add a random brightness factor to the training image and also some noise to the training angle. This helps the neural network generalize better and we can see that it performs much better on the validation data which doesn't have this kind of noise.

Flipping the images on the vertical axis is another very common technique used in Convolutional Neural Networks that is used by me here. This gives the network orientation invarianve. I, in turn multiply the steering angle by -1. This serves like the idea presented in the lectures on the lines of driving the car in the opposite direction.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I trained the model for just 15 epochs and it gave me close to perfect driving on track 1. I used an adam optimizer so that manually tuning the learning rate wasn't all that necessary.

The following image gives a sense of the progression of the training based on the training and validation losses.

![alt_text][image3]
