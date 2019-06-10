# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./WriteUpImages/Dist1.png "Distribution"
[image2]: ./WriteUpImages/Dist2.png "Distribution 2"
[image3]: ./WriteUpImages/Dist3corrected.png "Distribution corrected"
[image4]: ./WriteUpImages/dist3.png "Distribution corrected"
[image5]: ./WriteUpImages/histo.png "Histogram"
[image6]: ./WriteUpImages/Stop.png "Stop"
[image7]: ./WriteUpImages/70.png "Stop"
[image8]: ./WriteUpImages/kids.png "kids"
[image9]: ./WriteUpImages/kidssun.png "kidssun"
[image10]: ./WriteUpImages/left.png "left"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](Traffic_Sign_Classifier_final.ipynb)


### Data Set Summary & Exploration

#### 1. Load the Data
First section will load the Data sets with pickle. There are three sets that will be handled, Trining Set, Valid Set and the Test Set. The main boundaries of the Data Set is also printed. This shows the number of trining examples, number of test examples and the image shape with the number of classes used in the data set.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Visualize the Dataset
The next step ist do visualise the dataset. From every class, 7 images are shown. The number of images in each class is also plotted. This shows the distribution of all classes and it also shows, that a few classes just have little data in it.
* Max. number of images in class: 2010
* Min. number of images in class: 180
* Mean of all images in class: 809

![alt text][image1] ## Bild einfügen!!!

#### 3. Preparing the training set
As seen in the distribution in the classes, there are a few classes with less data. This will lead to an underrated model for this classes. Therefore, the classes are prepared with an RandomOverSampler algorithm. All classes will be over sampled to the max. amount of classes available (2010 classes).

![alt text][image2]
![alt text][image3]

#### 4. Equalizing the images
Since the images in the data set are very inconsistent, the images have to be equalised. The lighting for the images differs from image to image and also the texture. Therefore, the Y channels of each image are equalised with the cv2.equalizeHist function. This is performed for all data sets and gives an output of the images as following:
![alt text][image5]

The following distribution shows the images in the three data sets. As seen, the training data is now all at 2010 images per class:
![alt text][image4]


### Design and Test a Model Architecture

#### 1. Seting up the model

Before the data is given to the model, a last preprocessing step is used. The data is normalises and shuffled for each set.

#### 2. Model achrchitecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling		| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x16     									|
| RELU		|        									|
| Max pooling		| 2x2 stride,  outputs 5x5x16 				|
| Flatten		|  				|
| Fully connected		| Input = 400, Output = 120 				|
| RELU		|        									|
| Fully connected		| Input = 120, Output = 84 				|
| RELU		|        									|
| Fully connected		| Input = 84, Output = 10 				|



#### 3. Model training

All parameters are used from LeNet network. The only exception is the used Epochs. This is currently set to 50 but it's also shown, that it's not converging any further so it can be lower to reduce calculation time.

#### 4. Results of the training

My final model results were:
* training set accuracy of 1
* validation set accuracy of 0.958 
* test set accuracy of 0.944

If a well known architecture was chosen:
* What architecture was chosen?
The network is working with LeNet 5.
* Why did you believe it would be relevant to the traffic sign application?
LeNet shows some good results for image classification and its super fast and accurate.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The over all accuracy with the ok size validation and testing set looks promising. If the model really performs well, has to be proven later.



### Test a Model on New Images

#### 1. Feeding with new images

A new image pipeline is defined with the correct labels. Here is an example of the image choosen:

![alt text][image6]
This image looks easy to detect. The angle of the image is not the best but it's clearly visuable and the shape is detectable.

![alt text][image7]
This image is perfect for detection. The background is an even coler and there is nothing much that can disturb the image.

![alt text][image8]
This image is very dark and harder to detect. The pre processing algrorithem is very important for that image and will correct a few mistakes.

![alt text][image9]
The same applies for this image. The sun creats some very bright spots.

![alt text][image10]
This image is nothing special and should be no problem for the network.

#### 2. Model Prediction

Here are the results of the prediction:

* 01.png: ERROR detected "Priority road", actual "Stop"
* 002.png: correctly identified "Speed limit (70km/h)"
* 003.png: correctly identified "Speed limit (70km/h)"
* 004.png: correctly identified "Stop"
* 005.png: correctly identified "Stop"
* 006.png: correctly identified "Stop"
* 007.png: correctly identified "Stop"
* 008.png: correctly identified "Children crossing"
* 009.png: correctly identified "Children crossing"
* 010.png: ERROR detected "Road work", actual "Children crossing"
* 011.png: ERROR detected "Right-of-way at the next intersection", actual "Children crossing"
* 012.png: ERROR detected "Wild animals crossing", actual "Speed limit (30km/h)"
* 013.png: correctly identified "Yield"
* 014.png: correctly identified "General caution"
* 015.png: correctly identified "Speed limit (30km/h)"
* 016.png: ERROR detected "Priority road", actual "Stop"
* 018.png: correctly identified "Stop"
* 020.png: ERROR detected "Speed limit (80km/h)", actual "Speed limit (100km/h)"
* 021.png: correctly identified "No vehicles"
* 022.png: ERROR detected "Stop", actual "Speed limit (80km/h)"
* 023.png: ERROR detected "Stop", actual "Vehicles over 3.5 metric tons prohibited"
* 024.png: ERROR detected "Bicycles crossing", actual "General caution"
* 025.png: ERROR detected "Speed limit (50km/h)", actual "Speed limit (100km/h)"
* 026.png: ERROR detected "Speed limit (50km/h)", actual "Speed limit (80km/h)"
* 028.png: correctly identified "Keep left"
* 029.png: correctly identified "Keep left"
* 032.png: correctly identified "Stop"


The model does not work as good as thought it wold. It is struggling with images on a dark background. This issue can be solved by preprocessing the images or by training with more images which contain a black background.

#### 3. Softmax

Since 32 Images are used to test the program, only the first 6 softmax functions are shown below:

As shown a both, the first image was classified wrong and you can see, that its also struggling with the prediction. Three predictions are very close together. All other functions are quite clear and the network is not struggling at all.

Image 1:  
Stop  
  0.45151 Priority road  
  0.26656 Bicycles crossing  
  0.26523 No entry  
  0.00901 Traffic signals  
  0.00295 Stop  
Image 2:  
Speed limit (70km/h)  
  1.00000 Speed limit (70km/h)  
  0.00000 Go straight or right  
  0.00000 Speed limit (80km/h)  
  0.00000 Turn right ahead  
  0.00000 Speed limit (20km/h)  
Image 3:  
Speed limit (70km/h)  
  1.00000 Speed limit (70km/h)  
  0.00000 Speed limit (30km/h)  
  0.00000 No vehicles  
  0.00000 Speed limit (80km/h)  
  0.00000 Stop  
Image 4:  
Stop
  0.99998 Stop  
  0.00002 No vehicles  
  0.00000 No passing  
  0.00000 No entry  
  0.00000 Speed limit (50km/h)  
Image 5:  
Stop  
  0.98342 Stop  
  0.00954 No vehicles  
  0.00704 No entry  
  0.00000 Speed limit (70km/h)  
  0.00000 Bumpy road  
Image 6:  
Stop  
  0.99893 Stop  
  0.00107 No vehicles  
  0.00000 No entry  
  0.00000 No passing  
  0.00000 Priority road  

## How to improve the network

The first step to get better results would not be to change the network architecture. The network should work fine as it is design right now. More important would be the image pre processing. In this process, the results of the network can be improved a lot. It would be interesting to see how the networks performed with different input images (Only Grayscale, only normalised, ...)

