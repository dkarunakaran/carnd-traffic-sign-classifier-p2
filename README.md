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

**Model Architecture**
```
Input

The LeNet architecture accepts a 32x32x3 image as input

Architecture

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

Output
Return the result of the 3rd fully connected layer.
```
Code look like below:

```
def LeNet(x): 
    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = 0, stddev = 0.1))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    
    # Activation 1.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    
    # Layer 2: Convolutional. Input = 14x14x6. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = 0, stddev = 0.1))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation 2.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Flatten. Input = 5x5x16. Output = 400.
    flattened   = flatten(conv2)
    
    #Matrix multiplication
    #input: 1x400
    #weight: 400x120 
    #Matrix multiplication(dot product rule)
    #output = 1x400 * 400*120 => 1x120
    
     # Layer 3: Fully Connected. Input = 400. Output = 120.
    fullyc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = 0, stddev = 0.1))
    fullyc1_b = tf.Variable(tf.zeros(120))
    fullyc1   = tf.matmul(flattened, fullyc1_W) + fullyc1_b
    
    # Full connected layer activation 1.
    fullyc1    = tf.nn.relu(fullyc1)
    
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fullyc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = 0, stddev = 0.1))
    fullyc2_b  = tf.Variable(tf.zeros(84))
    fullyc2    = tf.matmul(fullyc1, fullyc2_W) + fullyc2_b
    
    # Full connected layer activation 2.
    fullyc2    = tf.nn.relu(fullyc2)
    
    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fullyc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = 0, stddev = 0.1))
    fullyc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fullyc2, fullyc3_W) + fullyc3_b
    
    return logits

```

To train the model, I used following hyperparameter after several trial and error method.

```
#Hyper parameters
learning_rate = 0.001

epochs = 1 #30

batch_size = 32
```
Lenet architecure gives the logits and cross entropy and loss operation gived the error compared to actual result and predicted result. Adamoptimiser is used to minimize the error.

```
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss_operation)
```
The above steps do the forward and backward pass and doing this on iterative manner will reduce the error at the end. The training code look like below where you have control over epoch, learning_rate, and batch size.

```
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        valid_loss, valid_accuracy = evaluate(X_valid, y_valid)
        print("Epoch {}, Validation loss = {:.3f}, Validation Accuracy = {:.3f}".format(i+1, valid_loss, valid_accuracy))
        print()
        
    saver1.save(sess, './classifier')
    print("Model saved")
```
Evaluate function in the above code will give the validation accuracy at each epoch.

```
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver1 = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        loss, accuracy = sess.run([loss_operation, accuracy_operation], feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
    return total_loss/num_examples, total_accuracy/num_examples
```

The higest validation accuracy reached around 94.5 and test validation accuracy at 93.5.




