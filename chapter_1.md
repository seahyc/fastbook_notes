# Chapter 1
#fastbook

## General Definitions
* **Machine learning**: training of a computer program to learn from its experience, instead of expert-crafted rules
* **Neural networks**: Network of artificial neurons where the nodes of the network are modelled as weights (+ is excitatory, - is inhibitory), and all inputs are modified by weights and summed in a linear combination for predictions
* **Deep learning**: ML technique using layers of neural networks, where outputs of previous layers are inputs for the next, and layers are trained by algorithms which minimise their errors in predictions over epochs
* **Universal approximation theorem**: proof that neural networks can solve problems to any level of accuracy 
* **Stochastic gradient descent**: General way to update weights of a neural network for improvement at any given tasks
* **Model performance**: Accuracy of the model at predicting the correct answers
* **Loss**:  A measure of model performance *(loss guides the training system during training, metrics is for humans to judge model performance after training, during validation)*
* **Overfitting**: Fitting the model too closely to the training data, to the extent it impairs predictions of unseen data; can think of it as the opposite of overgeneralisation
* **Metrics**:  Measurement of the model performance against the validation set, commonly chosen ones are *error rates* (% of wrong predictions as vetted against the validation set) and *accuracy* (1.0 - error rates)
* **Epoch**: 1 training cycle in which the model is run with the entire training data set as input once
* **Head**: In transfer learning, the last layer of the pre-trained model is replaced by this layer called the head that is customised for the task at hand
* **Architecture**: A mathematical function which takes in input and parameters
* **Model**: Architecture + parameters
* **Parameters**: Values in model that affects what tasks it can do, and is updated by model training
* **Fit**: Synonym with training, updating the parameters to decrease loss of a model
* **Fine-tune**: Adapting a pre-trained model to a slightly different task
* **Segmentation**: A sub-task of image recognition to recognise objects in an image
* **Batch size**: Number of samples processed before the model is updated
* **Tensors**: Colloquially, eneralisations of matrix to N-dimensional space
* **Test set**: Even more reserved subset of data to validate our choices of hyper parameters as model builders


## Limitations of Machine Learning
* Labelled data is required
* Can only learn from the data provided
* Only can make predictions, not recommendations, and so problems have to be framed as such
* Potentially result in unintended positive feedback loops, depending on how their results are interpreted and acted upon

## Some Practitioners’ Tips:
* Picking the architecture isn’t that important, as there are some popular choices that work well for most use cases (eg. ResNet)
* Transfer learnings is an understudied academic field, so the baseline resources required for deep learning applications might be overestimated
* Non-image problems can be converted into CV problems through visualisation of non-visual data, a fair bit of the practitioner’s expertise lies in problem framing
* Rule of thumb of image recognition: if human eye can tell the difference, most likely a deep learning model can too
* Pre-trained models for tabular data aren’t as common in the wild, they are typically proprietary
* Prototype with subsets of data set, before scaling out to the full set
* In choosing data for the test set, choose data that is representative of new data (eg. latest data in a time series, or entities not in training data set)

## Code Explanation (via comments)

### Image Classifier for cats, using Image Data Loaders

```
from fastai2.vision.all import *
path = untar_data(URLs.PETS)/‘images’

def is_cat(x): return x[0].isupper()

# valid_pct is the % of the data to be randomly split into validation set, seed allows that split to be reproduced, label_func transforms the file names, item_tfms applies transformation functions to each sample

dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))


# builds a cnn learner with data and the architecture
learn = cnn_learner(dls, resnet34, metrics=error_rate)

# fine tunes it for 1 epoch
learn.fine_tune(1)
```


### Training image segmentation model with camvid dataset, using segmentation data loader
```
# decompress dataset from a url 
path = untar_data(URLs.CAMVID_TINY)

# Create from list of fnames (filenames) in path with label_func applied; bs is batch size, which is the number of samples processed before the model is updated
# [fastai2/data.py at master · fastai/fastai2 · GitHub](https://github.com/fastai/fastai2/blob/master/fastai2/vision/data.py#L172)

dls = SegmentationDataLoaders.from_label_func(
    path, bs=8, fnames = get_image_files(path/“images”),
    label_func = lambda o: path/‘labels’/f’{o.stem}_P{o.suffix}’,
    codes = np.loadtxt(path/‘codes.txt’, dtype=str)
)

# U-Net is CNN developed for biomedical image segmentation, this function buildes a unet learner from data and architectured
learn = unet_learner(dls, resnet34)
# the first argument is the number of epochs, the 2nd is the base learning rate
learn.fine_tune(8)
```

### Training movie sentiments analysis model, with text data loaders

```
from fastai2.text.all import *

# samples in test folders are kept for validation testing

dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')

# building text learner, with the data set and awd_lstm architecture, drop_mult is a global multiplier that scales dropouts in LSTM, dropout is the probability of recurrent connections and inputs being excluded from activation and weight updates, to reduce overfitting

learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

# train for 4 epochs, learning rate is 0.01
learn.fine_tune(4, 1e-2)

```

### Training tabular learner, with tabular data loaders

```
from fastai2.tabular.all import *
path = untar_data(URLs.ADULT_SAMPLE)

# loads data from csv file, y_names identifies the dependent variable, cat_names identify categorical independent variables, cont_names for continuous ones, procs is an array of processes applied on the data set

dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",
    cat_names = ['workclass', 'education', 'marital-status', 'occupation',
                 'relationship', 'race'],
    cont_names = ['age', 'fnlwgt', 'education-num'],
    procs = [Categorify, FillMissing, Normalize])

learn = tabular_learner(dls, metrics=accuracy)
```

### Training learner for collaborative filtering

```
from fastai2.collab import *
path = untar_data(URLs.ML_SAMPLE)
dls = CollabDataLoaders.from_csv(path/'ratings.csv')

# y_range is applies a sigmoid function from 0.5 to 5.5

learn = collab_learner(dls, y_range=(0.5,5.5))
learn.fine_tune(10)
```


## Questionnaire

1. Do you need these for deep learning?
	* Lots of math T / **F**
	* Lots of data T / **F**
	* Lots of expensive computers T / **F**
	* A PhD T / **F**
2. Name five areas where deep learning is now the best in the world.
	1. Computer vision
		1. Image generation
		2. Medicine
		3. Biology
	2. Natural language processing
	3. Recommendation systems
	4. Playing games
	5. Robotics



3. What was the name of the first device that was based on the principle of the artificial neuron?
*Mark I Perceptron*

4. Based on the book of the same name, what are the requirements for parallel distributed processing (PDP)?
	1. Processing units
	2. State of activation
	3. Output function for each processing unit
	4. Patterns of connectivity between the processing units
	5. Activation rule for how the processing units combine the inputs and its current state to produce an output
	6. Propagation rules for how the patterns of activity are propagated through the network of connectivity
	7. Learning rule for how the pattern of connectivity are modified by new experience
	8. Environment in which the system operates

5. What were the two theoretical misunderstandings that held back the field of neural networks?
	1. A single layer of neurones will not allow the device to learn simple but critical mathematical functions
	2. Adding more layers of neurones will improve the performance of the network, but these networks were too slow and big

6. What is a GPU?
	1. A graphical processing unit (aka graphics card) is a special processor that can conduct multiple (hundreds of thousands)  tasks in parallel

7. Open a notebook and execute a cell containing: 1+1. What happens?
	1. 2

8. Follow through each cell of the stripped version of the notebook for this chapter. Before executing each cell, guess what will happen.

9. Complete the Jupyter Notebook online appendix.

10. Why is it hard to use a traditional computer program to recognize images in a photo?
	1. It is nearly impossible to hard code all the features that go into recognising a nearly infinite kinds of images, many of which we aren’t even consciously aware of ourselves

11. What did Samuel mean by “weight assignment”?
	1. These are the setting of values for variables that define the way the program works

12. What term do we normally use in deep learning for what Samuel called “weights”?
	1. Parameters

13. Draw a picture that summarizes Samuel’s view of a machine learning model.
	1. inputs -> models -> results

14. Why is it hard to understand why a deep learning model makes a particular prediction?
	1. Because it is difficult to deconstruct the intermediate decisions that lead to the final prediction, and it is hard to tell how a model will predict based on data different from that it was trained on

15. What is the name of the theorem that shows that a neural network can solve any mathematical problem to any level of accuracy?
	1. Universal approximation theorem

16. What do you need in order to train a model?
	1. Training data set

17. How could a feedback loop impact the rollout of a predictive policing model?
	1. It might predict certain districts to be higher risk of crimes, prompting the police force to police that area more heavily, surfacing more crimes, and thus reinforcing the model to highlight that area again

18. Do we always have to use 224×224-pixel images with the cat recognition model?
	1. Not necessarily, that was just the standard size historically for pre-trained models

19. What is the difference between classification and regression?
	1. Classification is for predicting categorical data, whilst regression is for predicting continuous data

20. What is a validation set? What is a test set? Why do we need them?
	1. A validation set is meant to evaluate the accuracy of a model trained on a training data set; a test set is an even more highly reserved data set that even the model builder cannot see, to avoid overfitting hyper-parameters.

21. What will fastai do if you don’t provide a validation set?
	1. It will randomly parcel out by default 20% of the data set for validation

22. Can we always use a random sample for a validation set? Why or why not?
	1. Not always, because the validation set might not be representative of new data. eg. most recent time for time series, or new entities in validation

23. What is overfitting? Provide an example.
	1. Overfitting means your model is too sensitive to the peculiarities of your training data set, and does not generalise well to new data. An example is an image recogniser for cats identifying a particular colour of fur as a feature

24. What is a metric? How does it differ from “loss”?
	1. A metric is a number that tells the model builder how many labels the model correctly predicts in the validation set. It is meant for human consumption, and not as an input for model training as loss is.

25. How can pretrained models help?
	1. It reduces the time and computational resource required for training, and the data set required

26. What is the “head” of a model?
	1. This is the layer that replaces that last layer of a pre-trained model, that is adapted to the specific task at hand

27. What kinds of features do the early layers of a CNN find? How about the later layers?
	1. Earlier layers find more foundational features, such as simple lines or curves, while the later layers identify more sophisticated features that are built up from the earlier layers, such as a a shape, or even facial features

28. Are image models only useful for photos?
	1. Not necessarily, any domain that can be visualised can be tackled with an image model

29. What is an “architecture”?
	1. Mathematical model which takes in the inputs and parameters

30. What is segmentation?
	1. Recognising localised objects within an image

31. What is y_range used for? When do we need it?
	1. y_range is used for specifying the range of continuous values the model predicts. It is used when the output is a continuous value

32. What are “hyperparameters”?
	1. Parameters about parameters, they influence the meaning of parameters (eg. architecture choices, batch sizes)

33. What’s the best way to avoid failures when using AI in an organization?
	1. Always setting aside a test set that the model builders don’t have access to, setting metrics for a test set up front, and determining baselines for those metrics
