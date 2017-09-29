#**Traffic Sign Recognition** 

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This document forms part of my submission. I have included my associated project files in a zip file rather than on Github

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I didn't use numpy or panda to obtain the properties of the dataset. In hindsight I think pandas would have made the sorting of the data easier, I had to do it manually using arrays and it was frustrating to get to work.
I used the .shape[1:] poperty of the sets to obtain the shape of the images
I used dataset_set.shape[0] to get the size of each set. I could have also used len(dataset_set) to find the number of examples in a set.
I used the np.unique() method, passing the y labels to it in order to achieve the number of unique labels.

* The size of training set is 27839
* The size of the validation set is 6960
* The size of test set is 12630
* The shape of a traffic sign image is 32,32,3
* The number of unique classes/labels in the data set is 43
* Min number samples per class: 180
* Max number samples per class: 2010.0


I printed a summary of the count of each class in ascending order. This helped to identify the classes that had the fewest examples.
The following are the classes with less that 500 examples in the training set.

* Class: 0	  Count: 180 	 Name: Speed limit (20km/h)
* Class: 37	  Count: 180 	 Name: Go straight or left
* Class: 19	  Count: 180 	 Name: Dangerous curve to the left
* Class: 32	  Count: 210 	 Name: End of all speed and passing limits
* Class: 27	  Count: 210 	 Name: Pedestrians
* Class: 41	  Count: 210 	 Name: End of no passing
* Class: 42	  Count: 210 	 Name: End of no passing by vehicles over 3.5 metric tons
* Class: 24	  Count: 240 	 Name: Road narrows on the right
* Class: 29	  Count: 240 	 Name: Bicycles crossing
* Class: 39	  Count: 270 	 Name: Keep left
* Class: 21	  Count: 270 	 Name: Double curve
* Class: 40	  Count: 300 	 Name: Roundabout mandatory
* Class: 20	  Count: 300 	 Name: Dangerous curve to the right
* Class: 36	  Count: 330 	 Name: Go straight or right
* Class: 22	  Count: 330 	 Name: Bumpy road
* Class: 6	  Count: 360 	 Name: End of speed limit (80km/h)
* Class: 16	  Count: 360 	 Name: Vehicles over 3.5 metric tons prohibited
* Class: 34	  Count: 360 	 Name: Turn left ahead
* Class: 30	  Count: 390 	 Name: Beware of ice/snow
* Class: 23	  Count: 450 	 Name: Slippery road
* Class: 28	  Count: 480 	 Name: Children crossing


####2. Include an exploratory visualization of the dataset.

This section provides an exploratory visualization of the data set. 
Firstly, I plotted an example of each class. I selected a random image from each class, this way as I worked I would get a different image for each class every time, allowing me to assess the quality and differences of the images as I worked

The images are generally quite unclear, with poor contrast. I may try to address improve the contrast through processing if the results are not satisfactory and time permits.
After reviewing the signs, I can imagine that the speed limit signs in particular may be difficult to classify, and the low resolution and poor contrast means that it is difficult to discriminate between them.
In addition, due to the uneven distribution of number of samples of the speed limit classes, it is even more likely that the speed limits signs with the lowest sample size are likely to be misclassified

In addition, it is worth pointing out that in some images the signs are partially obscured by vehicles other parts of the scene.

![Training Set Samples](writeup_images\trainingsetvis.png)
I plotted a bar chart for each set showing the distribution of examples per set.
![Training Set Samples](writeup_images\bar_training_org.png)
![Validation Set Samples](writeup_images\bar_valid_org.png)
![Test Set Samples](writeup_images\bar_test_org.png)
The examples are not evenly distributed across the classes in the sets. This is likely to have a detrimental effect on training.
The likely outcome will be that the classifier will not be able to generalise for images of the classes that have a low number of instances in the training set.
Later in the task I plan to try to resolve this by creating new data to try to balance the training dataset.


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


Initially I left the images as colour, as my original thoughts was that the colour might help to classify the images correctly, given that a number of the signs have different base colours.
My preprocessing involved normalising the images so that their values are approximately centered around zero, and they have a value range of greater than -1 and less than 1.
To do this I subtracted 128 (shifting the intensity levels by half of the range of values of the intensity level, ie half of the range of values of a byte) and deviding by 128. I suspect that there are better ways of doing this, but this resulted in a mean of approximately -0.35, a min of 0.97 and a max of 0.99. Note these values may have changed due to the augmentation of the dataset following at a later point.
This confirms that it is centred approximately at 0 and between -1 and 1.
Normalising the data should make the training process more efficent, as the optimiser will not have to search as hard to minimise the errors.

I did also consider using max min scaling, resulting in a mean of +0.3 and a min of 0 and a max of 1. I dismissed this as I was satisfied with the performance using the former method.



Here is an example of a traffic sign image before and after grayscaling, and after normalisation.
![Sample Colour, GrayScale and Normalised Images](writeup_images\normalisation.png)




To increase the amount of data in the dataset I had the following ideas.

1. Mirror any signs that can be mirrored and not change their labels
2. Mirror any signs that can be mirrored and swap the labels with the appropriate class. (Such as left and right arrows)
3. Rotate the images
4. Warp the images


Unfortunately, I had a bug in items 1 and 2 that caused the labels to get mixed up and I was unable to resolve, so I implemented parts 3 and 4 instead.

Before I started the augmentation, I first determine how many new samples to create. I initially did this by looking that the original dataset, and seeing that the highest instance of classes was 2010. I decided that I would create enough data so that every class has 2500 instances. This meant a total training set of 107500 examples.
However, after trials, it appeared that this may have been having a negative effect on the performance of the classifier, so I tried 2 further options, firstly creating enough augmented images to double the count in each class. Secondly, I tried creating a fixed number of augmented images to supplement the set. I settled on using the fixed number of images, which resulted in a smaller dataset, but had a lower ratio of fake to real images. The benefit of the smaller dataset was that it decreased training time, which providing the objectives could be satisfied still was useful.

I rotated the images by random values in the range +/- 10 degrees, and added it to a X_train array of images, and added the class to the y_train array.
In order to warp the images, I used the open cv warpPerspective() function. This function requires that you provide 2 sets of points. The image is then warped by stretch the image by moving the first set of points to the second and translationg the image proportionally to this. I randomly generated 2 sets of 4 points. The random points we generated so that the x values were within 6 pixels of the image boundary, and the y values within 4. My reason for this was that I believed that this will skew the x axis more that the y, which will give a perspective change similar to driving past a sign. In the real world the perspective changes along the horizontal axis more that the vertical axis as a vehicle passes a sign
By randomly generating both sets of points, it meant that the image can be stretched or compressed.
I repeated the process for each image in each class in a loop until I had enough images to satisfy my original goal.

Below is a sample augmented image of 2 class class.
![Augmented Test Set Images](writeup_images\augmented_images_1.png)
![Augmented Test Set Images](writeup_images\augmented_images_1i.png)

Below is a bar graph of the final test set.
![Augmented Test Set](writeup_images\augmented_ds_fixed.png)


Note I did not resize the validation and test sets like I suspect that I should have, because I did not want the augmented data being included in these sets. I do not know if this was the correct thing to do.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							|
| Convolution	     	| 1x1 stride, valid padding, outputs 28x28x64 	|
| RELU					| RELU as activation 							|
| Dropout				| Dropout with Keep Prob set to 0.5 during training|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution	    	| Output 10x10x16 								|
| RELU					| RELU as activation							|
| Dropout				| Dropout with Keep Prob set to 0.5 during training	|
| Max pooling	      	| 10x10x16. Output = 5x5x16						|
| Flatten				|	Input = 5x5x16. Output = 400				|
| Fully Connected		|	Input = 400. Output = 120					|
| RELU					| RELU as activation							|
| Dropout				| Dropout with Keep Prob set to 0.5 during training	|
| Fully Connected		|	Input = 120. Output = 84					|
| RELU					| RELU as activation 							|
| Fully Connected		|	Input = 84. Output = 43						|




Following this, I used top k and softmax. Top k returns both the softmax and indices of the requested number of top predictions.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
I used the Adam Optimizer as used in previous exercises in the course so far. As this had worked satifactoriliy previously I didnt feel that m knowledgewas sufficient to justify changing to an alternative.
As I trained on my laptop rather that Amazon EC2, I kept the batch size to be reasonably small, 100.
For the Epochs, I iteratively trialled a number of values, before I finally tried 50.
I originally set the learning rate to 0.001, I trailed different values in the range of 0.05 to 0.0001. 
I found that 0.001 actually performed better than other many other values, so I reverted back to this. There was not a significant performance difference between 0.001 and 0.007



####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Validation Accuracy = 0.960
* Test Accuracy = 0.939
* Training Set Accuracy = 0.978
* Accuracy on additional images = 1.000

The result above are satifactory. The target of > 93% accuracy against the validation set has been satisifed.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
My approach was iterative to a certain extent. It was based upon the exercises that I had completed thus far in the course. Once I had the initial architecture implemented, I modified the hyperparameters in order to achieve a suitable performance. However, I found that this approach was very time consuming, an the discovery of bugs during the process meant that previous baselines in performance we no longer relevant as they had been gained using erroneous software.
After modifying the parameters considerably, the performance plateaued. In addition, the performance on my additional images from online was very poor. At this point I introduced dropout to prevent over fitting, with the hope that it may generalise better on the new images.




##### Below are some of the steps in the iterative process:
(Note I have not listed all the iterations as there were many, and some were discarded due to finding errors in my software after running.

I originally used the exact same implementation from the LeNet exercise in the Conv Net module. As described in the videos this is a good starting point.
In the prior exercise, I used only a small number of Epochs, 10, and a learning rate of 0.001.
However, I decided to initially start with full colour images, meaning that the input shape was 32x32x3 to begin with. 
As discussed previously this was because I believed that the colour would be useful.

The following test cases were run:

Epochs: 10, Learning Rate 0.001:
* Validation Accuracy = 0.905
* Test Accuracy = 0.705

Note during this attempt I had not yet implemented the section with my own images.
Clearly not good enough. However, it was clear from my output that the accuracy was still increasing at a good rate, and that increasing the number of Epochs would result in a improved accuracy.



Increasing the Epochs to 50 gave the following results

Epochs: 50, Learning Rate 0.001:
* Validation Accuracy = 0.921 ( was 0.948 at Epoch 48)
* Test Accuracy = 0.730
* Training Set Accuracy = 0.977
* Additional images = 0.200



At this stage I adjusted the training rate to 0.0005, with Epochs still at 50.
This give an improvement to validation accuracy but test accuracy remained unchanged
Epochs: 50, Learning Rate 0.0005:
* Validation Accuracy = 0.940 ( was 0.948 at Epoch 48)
* Test Accuracy = 0.730
* Training Set Accuracy = 0.977 
* Additional images = 0.200

There was no significant improvement on the Test set, so at this point I needed to look at alternative methods to optimise.



 Pierre Sermanet and Yann LeCun's paper on Traffic Sign Classification describes how using grayscale images yielded an improvement in performance. Therefore I modified my code to convert to grayscale.
Fortunately this is a simple enough exercise with Tensorflow, requiring that the shape of the placeholder and input depth are changed.
LeCun and Sermanet used a variant of the LeNet architecture

Changing to grayscale made a huge improvement.
The figures were now:
Epochs: 50, Learning Rate 0.0005, grayscale input:
* Validation Accuracy = 0.980
* Test Accuracy = 0.882
* Training Set Accuracy = 0.944
* Additional images = 0.600



Up until now, my model has no regularisation. I decided that dropout could be worth trying in order to prevent overfitting.
I added dropout between each layer, with a keep probability of 0.5 during training.
This gave the results:
Epochs: 50, Learning Rate 0.0005, grayscale input, dropout keep_prob 0.5:
* Validation Accuracy = 0.973
* Test Accuracy = 0.923
* Training Set Accuracy = 0.981
* Accuracy on additional images = 0.800

At this point, I was satisfying the project brief with a good accuracy across the board. 
Hence I still felt I could do more, so at this point I decided to generate the additional images as described earlier.



Epochs:50, learning rate = 0.0007, keep_prob = 0.5, grayscale input, dropoutkeep_prob = 0.5

* Validation Accuracy = 0.927
* Test Accuracy = 0.895
* Training Set Accuracy = 0.895
* Accuracy on additional images = 1.00

Note the Validation set performance has now dropped to less than the target of 93



Epochs:50, learning rate = 0.001, grayscale input, dropoutkeep_prob = 0.5

* Validation Accuracy = 0.912
* Test Accuracy = 0.909
* Training Set Accuracy = 0.918
* Accuracy on additional images = 1.00


![Validation Accuracy Plot](writeup_images\50ep lr0.001.png)

Note that although the accuracy on the additional images has increased, the performance on the other sets has decreased. This may be because I am augmenting my images too much, and there are now images in the training set that are too similar to each other despite belonging to different classes. I modified my augmentation logic, previously it was generating images in order to create 2500 samples per class, so I changed the code to create a fixed number of  incremental images in a set instead.

In hindsight, I moved to 50 Epochs too soon, which meant that later test runs took much longer to train.



In order to allow me to identify trends more quickly I changed to 20 Epochs

Epochs:20, learning rate = 0.001, grayscale input, dropoutkeep_prob = 0.5

* Validation Accuracy = 0.917
* Test Accuracy = 0.910
* Training Set Accuracy = 0.905
* Accuracy on additional images = 1.000

As expected the reduction in Epochs reduced the accuracy across all sets. The Addition images set was still 100% but the confidence level for each sign was reduced.



Increasing the keep prob of the dropout

Epochs:20, learning rate = 0.001, grayscale input, dropoutkeep_prob = 0.7

* Validation Accuracy = 0.949
* Test Accuracy = 0.939
* Training Set Accuracy = 0.965
* Accuracy on additional images = 1.000

Increasing the dropout keep prob improved performance. This suggests that previously the network was dropping too higher a percentage and was causing the network to be slow to train.
The network was still able to classify the images from the internet successfully.


Change of keep_prob to 0.8

Epochs:20, learning rate = 0.001, grayscale input, dropout keep_prob = 0.8

* Validation Accuracy = 0.960
* Test Accuracy = 0.939
* Training Set Accuracy = 0.978
* Accuracy on additional images = 1.000

![Validation Accuracy Plot](writeup_images\20ep kp0.8 lr0.001.png)

Note I did not modify mu or sigma values as I did not believe that this would change the performance.




* What were some problems with the initial architecture?
The architecture was not able to generalise sufficiently on new images.
A potential solution to this was to use drop out. I added dropout into the LeNet architecture, with a keep value of 0.5.
I adjusted the value of dropout during training but found that a keep_prob value 0.7 gave the best results.


The initial architecture had no major flaws, but there were some improvements that could be made.
- The model was unable to generalise sufficiently to the test  data, and later the images that I found online. With the images I found online the initial accuracy was 0% in some instances.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* I adjusted many parameters over the course of the project. 
	* I adjused Epochs many times.Increasig to 50 gave good performance, but the time penalty meant that it slowed down progress.
	* I changed the input shape to cope with grayscale images, and the output to support 43 classes.
	* I changed the learning rate many times. 

* The Convolution Neural Network works well for the traffic sign classification problem, and other computer vision problems as they are able to learn by themselves what features are important within an image.
After the important features are identified, the ConvNet is able to the evaluate what the overall collection of identified features represents.


If a well known architecture was chosen:

I chose the LeNet architecture from the previous CarND activities. The reason be
The reasons behind this were that Pierre Sermanet and Yann LeCun used a variant of the LeNet architecture to perform traffic sign classification and were able to acheive an accuracy of over 99%.
Due to my lack of experience I decided to start with the LeNet from the previous activities, with the ambition to try to modify to match the Pierre Sermanet and Yann LeCun implementation if time permitted (it didn't)
The LeNet architecture worked well on the handwriting classifier from previously. Given the image sizes, and fundamentals of the problems were similar it seemed like a reasonable starting point.


* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The final model achieved the project requirements. 


 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![additional signs](more_signs\120kph_32x32.jpg) ![additional signs](more_signs\general_warn.jpg) ![additional signs](more_signs\humps.jpg)
![additional signs](more_signs\right.jpg) ![additional signs](more_signs\slippery_road.jpg)

Note: Apologies for the small images.
The following section shows the original image, along with the predictions, in a larger size.

I originally found 10 images, including some that we general road scenes that included a sign. I extracted the relavant sections and made a new 32 x 32 image.
I deliberately chose the following signs as I believed that they may be challenging

120 speed limit:
Has a reasonable number of samples, but the similarity of the different speed signs and the clarity of the images may cause issues

General Warning:
This is one of a few signs that have the same triangle shape. It is very similar in some images to the traffic signals sign.

Bumpy Road:
Again another triangle sign. Generally the quality of images of this sign are very poor, and would not always be able to be determined by a human.

Turn right ahead:
I think that this one should be ok. The image quality of the examples is above average

Slippery Road:
Again, another triangle sign, with quite a detailed image in the centre. May cause an issue due to the similarity with the bumpy road, children crossing, bicycles etc.



####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 120 speed limit     		| 120 speed limit   									| 
| General Warning     			| General Warning 										|
| Bumpy Road					| Bumpy Road											|
| Turn right ahead	      		|Turn right ahead					 				|
| Slippery Road			| Slippery Road      							|

Resulting Accurancy = 1.000 (100%)

Visualisation
![Validation Accuracy Plot](writeup_images\more_signs_conf_vis.png)more_signs_conf_vis.png

The model was able to correctly guess all traffic signs, which gives an accuracy of 100%. This compares favourably to the accuracy on the test set which was less than 100%, but this is primarily due to the new image set size.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

See image in previous section for visualisation of the result.

For the first image ...

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| 120 kph (correct) 									| 
| .000     				| 100 kph 										|
| .000					| 70 kph											|
| .000	      			| 30 kph					 				|
| .000				    | 20 kph      							|


For the second image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .991         			| General Warning 									| 
| .003     				| Bumpy Road 										|
| .003					| Pedestrians												|
| .002	      			| Traffic Signals					 				|
| .000				    | Road Narrows on right      							|


For the third image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Bumpy Road 									| 
| .000     				| Bicycles crossing 										|
| .000					| Slippery road												|
| .000	      			| Road work					 				|
| .000				    | Traffic Signals      							|

For the fourth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Turn right ahead									| 
| .000     				| Roundabout mandatory										|
| .000					| Go straight or left												|
| .000	      			| Ahead only					 				|
| .000				    | Stop      							|

For the fifth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Slippery road									| 
| .000     				| Dangerous curve to the left										|
| .000					| Double curve												|
| .000	      			| Wild animals crossing					 				|
| .000				    | Road work	    							|


Overall the performance is very good. The classifier has 100% Certainty on 4 of the five images, with the remaining image having 99%.
Note that the softmaxes have been rounded to 3 decimal places, so the other predictions are close to 0 enough to appear as 0.000, but they are not 0 exactly.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Unfortunately I was not able to complete this section.
