# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./images/nvnet.png "Model Visualization"
[image2]: ./images/loss.png "Loss per Echo"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_runxin.md summarizing the results
* visualization.ipynb containing the plotting codes
* run1.mp4 the vidoe recording the training model's self-driving result

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
I am using an end2end deep learning architecture introduced by [nVidia](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).

The model is inplemented by Keras in model.py from line 97 to 109, which consisting of following layers,
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_2[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 43, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 8, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 6, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 4, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 8448)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           844900      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
```

#### 2. Attempts to reduce overfitting in the model
I am not using any dropout or maxpooling layers, but I keep the number of echos as low as 3.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 118-120 in model.py), where the split rate is 0.2. 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 141).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of center lane driving, recovering from the left and right sides of the road as well. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet, I thought this model might be appropriate while based on the tutorial's result, it was not the best option.

Then after some literitual searching, I decided to use the deep learning network introduced by [nVidia](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, one of them is near the end line, where one side the road is a bisection to a mud road. My first training model fails to modify the steer and drive into the mud road. 

To improve the driving behavior in these cases, I increase the number of training set by several more records, which is especially near the end line.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 97-109) consisted of a convolution neural network with the following layers and layer sizes as detailed in previous section.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![Model Visualization][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one, which is forward and use center lane driving most of the time.

Then I repeated this process on track two, while this time it is backward, I just turn around. Most of the time I drive in center lane, while I reach both right and left lanes on purpose to teach the model's reaction.

To augment the data sat, I also flipped images and angles.

After the collection process, I had 25200 number of data points. I then preprocessed this data by normlization and cropping.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the following image, where at the third echo, the validation and training loss come to equal.
![Loss per Echo][image2]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
