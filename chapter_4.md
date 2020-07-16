# Chapter 4
#fastbook

## Core Concepts
### Tensor
In python, a list, or list of lists, or list of lists of lists… technically, N-dimensional vector where N is (0, infinity). When N <= 2, we have special names for them: 
* rank-0: scalar
* rank-1: vector
* rank-2: matrix 

#### Shape of a tensor
Refers to all the lengths of its axes
#### Axes of a tensor
Roughly translates to dimensions (different from the spatial dimensions)
#### Rank
Length of a tensor’s shape, or number of axes

### Numpy arrays
Multi-dimensional array of the same type, can be jagged, meaning all arrays can have different lengths
```
[
	[
		[0,0,0,5]
		[0,1,2]
	],
	[
		[0,0,0],
		[0,1,0]
		[0,1,2,1]
	]
]
```


### pytorch tensors
Multi-dimensional array of the same numeric type, have to be regularly shaped
```
[
	[
		[0,0,0]
		[0,1,2]
	],
	[
		[0,0,0]
		[0,1,2]
	]
]
```

### L1 norm
Mean of absolute value difference

### L2 norm
Root mean squared error, or the root of the mean of differences squared to make all differences positive

### Broadcasting
A pytorch shortcut that expands the smaller-ranked tensor to the same shape as the bigger-ranked one when conducting operations on the 2 tensors

### Backward propagation/ backward pass
Backward pass where the loss from that epoch is used to calculate the derivatives for each layer

### Forward pass
Computing the predictions from inputs

### Learning rate
Multiplier that is applied to the gradient to determine magnitude by which weights are updated

### Stochastic Gradient Descent (SGD)
An example of an optimiser function that helps to reduce loss over epochs

#### 7 steps in SGD
![](DA407CDB-179E-4FED-902D-A6E55F2D636D.png)

1. Initialise the weights (can be randomly)
2. Calculate predictions based on weights
3. Calculate loss (accuracy)
4. Calculate how the loss would change if weights change (gradient)
	1. Gradients are calculated for every weight, whilst holding the rest of the weights constant
5. Change all the weights in the calculated direction and magnitude
6. Repeat steps 2 to 5
7. Stop when model loss is good enough or time runs out

### Bias
A constant, the y-intercept

### Sigmoid function
Monotonic function that normalises input into an s-shaped curve, typically ranging from 0 to 1

### Batch
Group of samples over which loss of predictions made by a model is averaged over before updating the weights

### Mini-batch
A small group of samples, consisting of 2 lists, 1 of the inputs, and 1 of the labels (can think of as correct predictions, or the answer key).

### Dataset
A collection of dependent and independent variable pairings

### Rectified Linear Unit (ReLu)
Function that converts all negative values to 0.
 `f(x) = max(0, x)`

### Add in non-linearity
Simply adding up a bunch of linear functions just result in 1 linear functions, but if we want to model more complex functions, we have to introduce non-linearity in between the layers. Each layer is simply a linear function. The **universal approximation theorem** proves that by adding non-linearity into the equation, we can approximate any arbitrary mathematical functions. The intuition is that even arbitrarily complex graphs  can be approximated with shorter and shorter straight lines

### Activation function
Non-linear function, sometimes grouped together with linear functions in a layer.

### Parameters
In the context of artificial neural nets, these are the weights of the layers; they are numbers.

### Activations
The output of a neuron after passing its input through its activation function



## Practitioner's Tips
* Always calculate a **baseline performance** for your task with a simple model that performs reasonably well. This allows you to contrast the performance of more involved models and sees if the effort, time, costs scales well with increments in accuracy
* Baseline models can either be the simplest implementation you can think of (occam’s razor), or the standard solution out there in the wild
* L2 norm tends to exaggerate bigger differences due to the squaring
* Numpy arrays do not support gradient calculations and usage of GPU
* Good habit to check shapes of your tensors as you go, as a form of sanity check
* Take advantage of broadcasting in pytorch instead of standard python loops, as it doesn’t actually replicate the values, and operations are done in C or CUDA, so it’s much much more efficient
* In the MNIST example, accuracy not that helpful as a loss function when predicting categories, because it is a step function, meaning that gradient is mostly 0, and infinity at the threshold; the compromise is to use loss functions that are more sensitive to changes in weights
* Focus on accuracy instead of loss, since it more directly reflects what we want the model to do; loss is a compromise to facilitate automated training
* Use data loader to shuffle the composition of the batches to shake things up, and introduce some randomness in the training
* For deeper models (more layers), you can use lower learning rate and more epochs
* The deeper the model, the harder it is to optimise the parameters

## fastai API cheatsheet
* `path.ls()`
*Somewhat like unix ls, it tells you number of files in that path, and lists down all the files*

* `sorted()`
*Works on instantiation of L class, returns a list of files sorted by filename*

* `Image.open()`
*Image is from Python Imaging Library, most popular way of opening and manipulating images, already imported by Jupyter be default*

* `array()`
*returns numpy array*

* `tensor()`
*returns Pytorch tensor; can add [] after it and scope a sublist for each vector that you want returned. Position in the bracket indicates the n-rank of the vector to return*

* `tensor().requires_grad_()`
*Computes the gradient value at a given parameter value*

* `torch.randn()`
*Returns a random number from a normal distribution with mean of 0 and variance of 1*

* `y.backward()`
*Computes the gradient of the tensor using chain rule*

* `torch.cat()`
*Concatenates tensors in a given dimension, defaults to 0*

* Pytorch methods suffixed with `_` indicates an in-place operation

## Code Explanations
```
# Assign a tensor to im3_t
im3_t = tensor(im3)

# Convert a subset of that into a pandas dataframe
df = pd.DataFrame(im3_t[4:15,4:22])

# ** destructures the dict, allowing you to pass in keyword argumentsl background_gradient sets the background gradient of cell according to data in each column, according to Matplotlib color map
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
```

## Guiding Questions
1. How is a grayscale image represented on a computer? How about a color image?
2. How are the files and folders in the MNIST_SAMPLE dataset structured? Why?
3. Explain how the “pixel similarity” approach to classifying digits works.
4. What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.
5. What is a “rank-3 tensor”?
6. What is the difference between tensor rank and shape? How do you get the rank from the shape?
7. What are RMSE and L1 norm?
8. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?
9. Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.
10. What is broadcasting?
11. Are metrics generally calculated using the training set, or the validation set? Why?
12. What is SGD?
13. Why does SGD use mini-batches?
14. What are the seven steps in SGD for machine learning?
15. How do we initialise the weights in a model?
16. What is “loss”?
17. Why can’t we always use a high learning rate?
18. What is a “gradient”?
19. Do you need to know how to calculate gradients yourself?
20. Why can’t we use accuracy as a loss function?
21. Draw the sigmoid function. What is special about its shape?
22. What is the difference between a loss function and a metric?
23. What is the function to calculate new weights using a learning rate?
24. What does the DataLoader class do?
25. Write pseudocode showing the basic steps taken in each epoch for SGD.
26. Create a function that, if passed two arguments [1,2,3,4] and ‘abcd’, returns [(1, ‘a’), (2, ‘b’), (3, ‘c’), (4, ‘d’)]. What is special about that output data structure?
27. What does view do in PyTorch?
28. What are the “bias” parameters in a neural network? Why do we need them?
29. What does the @ operator do in Python?
30. What does the backward method do?
31. Why do we have to zero the gradients?
32. What information do we have to pass to Learner?
33. Show Python or pseudocode for the basic steps of a training loop.
34. What is “ReLU”? Draw a plot of it for values from -2 to +2.
35. What is an “activation function”?
36. What’s the difference between F.relu and nn.ReLU?
37. The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?
38. How do we know the function in SGD?
