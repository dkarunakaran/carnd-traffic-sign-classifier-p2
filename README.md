## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

---

### Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


### Data Set Summary & Exploration

#### 1) Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

  I used the numpy and pandas library to calculate summary statistics of the traffic signs data set:

  ```
  ### Replace each question mark with the appropriate value. 
  ### Use python, pandas or numpy methods rather than hard coding the results

  # TODO: Number of training examples
  n_train = len(X_train)

  # TODO: Number of validation examples
  n_validation = len(X_valid)

  # TODO: Number of testing examples.
  n_test = len(X_test)

  # TODO: What's the shape of an traffic sign image?
  image_shape = X_train[0].shape

  # TODO: How many unique classes/labels there are in the dataset.
  n_classes = len(pd.Series(y_train).unique())

  print("Number of training examples =", n_train)
  print("Number of validation examples =", n_validation)
  print("Number of testing examples =", n_test)
  print("Image data shape =", image_shape)
  print("Number of classes =", n_classes)

  ```

  Result:
  ```
  Number of training examples = 34799
  Number of validation examples = 4410
  Number of testing examples = 12630
  Image data shape = (32, 32, 3)
  Number of classes = 43
  ```
#### 2) Include an exploratory visualization of the dataset.

Random training images displayed to go through the dataset using matplotlib

--show images--
  
Then a ploted a diagrm to show of count of each signs in training data set

--show image--

#### 3) Design and Test a Model Architecture

The LeNet-5 architecture is used to predict the traffic signs. Before the training processs, dataset needs to have basic preprocessing using normalisation, grayscale etc. I found out normalisation itself gives very good output result and did notuser other preprocessing techniques.

I did a reshiffling of the data so that it can increase the random nature of the datset.
```
from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)
X_test, y_test = shuffle(X_test, y_test)

```

Then did the normalisation to make sure he image data has been normalized so that the data has mean zero and equal variance.
```
#Nomralisation
X_train = (X_train-X_train.mean())/(np.max(X_train)-np.min(X_train))
X_valid = (X_valid-X_valid.mean())/(np.max(X_valid)-np.min(X_valid))
X_test = (X_test-X_test.mean())/(np.max(X_test)-np.min(X_test))
```

Image before and after normalisation are displayed here:

--show images--

##### Model Architecture

**Input**
The LeNet architecture accepts a 32x32x3 image as input

**Architecture**
Layer 1: Convolutional. The output shape should be 28x28x6.

Activation. Your choice of activation function.

Pooling. The output shape should be 14x14x6.

Layer 2: Convolutional. The output shape should be 10x10x16.

Activation. Your choice of activation function.

Pooling. The output shape should be 5x5x16.

Flatten. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using `tf.contrib.layers.flatten`, which is already imported for you.

Layer 3: Fully Connected. This should have 120 outputs.

Activation. Your choice of activation function.

Layer 4: Fully Connected. This should have 84 outputs.

Activation. Your choice of activation function.

Layer 5: Fully Connected (Logits). This should have 43 outputs.

**Output**
Return the result of the 3rd fully connected layer.



