#**Traffic Sign Recognition** 
In this project, I have used deep neural networks and convolutional neural networks to classify traffic signs on the roads. The trained model is tested on 5 new images and the predictions by the model are displayed.  

## Writeup:
Dataset for the Projects: [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

	Files in the directory
	1. Ipython notebook: Traffic_Sign_Classifier.ipynb
	2. HTML File: Traffic_Sign_Classifier.html
	3. Writeup: writeup.md

	Folders in the directory
	1. writeup_images: Images used in the writeup
	2. test_images: 5 German Traffic Sign images.
	3. Model: Trained model files.
**Goals:**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/training_data.jpg "Training Data Visualization"
[image2]: ./writeup_images/testing_data.jpg "Testing Data Visualization"
[image3]: ./writeup_images/validation_data.jpg "Validation Data Visualization"

[image4]: ./writeup_images/data.jpg "Data"
[image5]: ./writeup_images/pre_processed_data.jpg "Preprocessed Data"

[image6]: ./writeup_images/lenet.png "Model"

[image7]: ./test_images/100_1607.jpg "Traffic Sign 1"
[image8]: ./test_images/left_turn.jpg "Traffic Sign 2"
[image9]: ./test_images/no_entry.jpg "Traffic Sign 3"
[image10]: ./test_images/Road_sign.jpg "Traffic Sign 4"
[image11]: ./test_images/triangle.jpg "Traffic Sign 5"

[image12]: ./writeup_images/1st.jpg "Softmax Probability for Image 1"
[image13]: ./writeup_images/2nd.jpg "Softmax Probability for Image 2"
[image14]: ./writeup_images/3rd.jpg "Softmax Probability for Image 3"
[image15]: ./writeup_images/4th.jpg "Softmax Probability for Image 4"
[image16]: ./writeup_images/5th.jpg "Softmax Probability for Image 5"

## Rubric Points
### 1. Writeup
###	2. Data Set Summary & Exploration
1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
2. Include an exploratory visualization of the dataset.

### 3. Design and Test a Model Architecture
1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.
2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

### 4. Test a Model on New Images
1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).
3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

### 5. Visualizing the Neural Network
---
###1. Writeup
- `writeup.md` is the writeup of the project in markdown format.

---

### 2. Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of test set is **12630**
* The size of the validation set is **4410**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

Code for above results is in **code cell 1st and 2nd** in the IPython Notebook.

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]
![alt text][image2]
![alt text][image3]

Code for above results is in **code cell 4th, 5th and 6th** in the IPython Notebook.

---

### 3. Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

### Pre-processing:
* Below are few samples of the data before pre-processing.

![alt text][image4]

* Code for above results is in **3rd code cell** in the IPython Notebook.
* For pre-processing, I have converted all theses images in the dataset to grayscale, applied Histogram equalization and later normalized them.

* I decided to convert the data to grayscale because performing operations on the 3 different channels(R, G, B) takes lot of processing power and time.
* Histogram equalization was done to get a clear image.
* I have done Normalization with zero mean to facilitate the convergence of the optimizer during training.
* I have also increased the brightness of every pixel by a factor of 1.2
* Below are the samples of the data after pre-processing.

![alt text][image5]

* Code for above results is in **code cell 7th, 8th and 9th** in the IPython Notebook.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is similar to the LeNet.
![alt text][image6]
Code of the LeNet_traffic architecture: **code cell 10**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input 1st Layer  		| 32x32x1 Pre-processed Image					|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Activation			| RELU  										|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6	 				|
| 						|												|
| Input 2nd layer		| 14x14x6										|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16	|
| Activation			| RELU        									|
| Max Pooling			| 2x2 stride,  outputs 5x5x16        			|
|						|												|
| Flatten				| output 400									|
|						|												|
| Fully Connected       | output 120									|
| Activation			| RELU											|
| Dropout1				| 0.95											|
|						|												|
| Fully Connected       | output 84									    |
| Activation			| RELU											|
| Dropout2				| 0.95											|
|						|												|
| Fully Connected       | output 43										|

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used following parameters:

	mu = 0
	sigma = 0.1
	learning rate = 0.001
	Optimizer = AdamOptimizer
	BATCH_SIZE = 64
	EPOCHS = 20
	dropout_rate = 0.95

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
Code: `code cell 15, 16, 17`

My final model results were:

* Training set accuracy: **99.80%**
* Validation set accuracy: **94.50%** 
* Test set accuracy: **92.00%**


**QUES 1:** What was the first architecture that was tried and why was it chosen?

**ANS:** First architecture I tried was the LeNet. LeNet works very well for image classification and it has been used since long time so I tried using LeNet first.  

**QUES 2:** What were some problems with the initial architecture?

**ANS:** Even after changing parameters and trying different pre-processing techniques, the accuracy wasn't increasing above 89%.

**QUES 3:** How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

**ANS:** To achieve the minimum required accuracy on the validation data, I added two dropout layers to the architecture. Using normalized data for training also increased the accuracy. Adding dropout layers had a soothing effect to the over-fitted model. 

**QUES 4:** Which parameters were tuned? How were they adjusted and why?

**ANS:** Parameters like `dropout_rate`, `BATCH_SIZE` and `EPOCH` were tuned to achieve desired level of accuracy. The model was over-fitted so dropout was added.

**QUES 5:** What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

**ANS:** There was a significant difference between the accuracy on training and the validation data. This signified the over-fitting of the model. For soothing the effect, dropout layers were added.


---

### 4. Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11]

* **Image 1:(Right-of-way at the next intersection)** The common feature is the triangle with other traffic signs so the model might get confused and detect other sign.
* **Image 2:(Turn Left Ahead)** Signs like 'turn right ahead' and 'ahead only' has common color features. So due to this, model might give an incorrect prediction.
* **Image 3:(No entry)** 'No passing' sign has common color features. Due to jittering and normalization, model might give an incorrect prediction.
* **Image 4:(Bumpy Road)** If the image is already bright enough then due to increasing brightness while pre-processing model might give an incorrect prediction.
* **Image 5:(Yield)** This image as distinctive property so prediction will be easy. Only in case of very low brightness and jittery image, the prediction will be incorrect.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way ... 		| Right-of-way ...								| 
| Turn left ahead		| No entry 										|
| No entry				| No entry										|
| Bumpy Road      		| Bumpy Road					 				|
| Yield					| Yield      									|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions is located in the **22nd code cell** of the Ipython notebook.

### For the first image:

![alt text][image12]

- For the first image, the model is relatively sure that this is a Right-of-way at the next intersection sign (probability of 84%), and the image does contain same sign. The top five soft max probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .84         			| Right-of-way at the next intersection			| 
| .15     				| Beware of ice/snow							|
| .00					| Slippery road									|
| .00	      			| Dangerous curve to the right	 				|
| .00				    | Pedestrians									|


### For the second image:

![alt text][image13]

- Turn left ahead has a max probability of 66% and it is correct. The top five soft max probabilities are:

 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .66         			| Turn left ahead								| 
| .26     				| Beware of ice/snow							|
| .06					| No entry 										|
| .00	      			| No passing					 				|
| .00				    | Dangerous curve to the left					|

### For third image:
![alt text][image14]

- No entry has a max probability of 80% and it is correct. The top five soft max probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .80         			| No entry  									| 
| .18     				| Turn right ahead								|
| .00					| No passing									|
| .00	      			| Priority road					 				|
| .00				    | Stop			    							|

### For fourth image:
![alt text][image15]

- Bumpy road ahead has a max probability of 99% and it is correct. The top five soft max probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Bumpy road  									| 
| .00     				| Speed limit (60km/h)							|
| .00					| Wild animals crossing							|
| .00	      			| Children crossing				 				|
| .00				    | Speed limit (20km/h) 							|

### For fifth image:
![alt text][image16]

- Yield has a max probability of 100% and it is correct. The top five soft max probabilities are:


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Yield		   									| 
| .00     				| Speed limit (60km/h)							|
| .00					| Turn left ahead								|
| .00	      			| Ahead only					 				|
| .00				    | Speed limit (50km/h) 							|


---

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


