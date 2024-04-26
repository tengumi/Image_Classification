# Image Classification

**Image Classification** is a fundamental task in vision recognition that aims to understand and categorize an image as a whole under a specific label. Unlike object detection, which involves classification and location of multiple objects within an image, image classification typically pertains to single-object images. When the classification becomes highly detailed or reaches instance-level, it is often referred to as image retrieval, which also involves finding similar images in a large database.

## About The Project

Welcome to this project, where the objective is to train an image classification model on a dataset of rice pictures. There are five distinct varieties of rice in this dataset. We will be using `convolutional neural networks (CNN)` to address this task. Additionally, we will be utilizing the `MobileNet v2` pre-trained model. Finally, we will compare the results of these two models to determine which one performed better.

## CNN

A `convolutional neural network` is a type of deep learning network used primarily to identify and classify images and to recognize objects within images.

### Architecture of convolutional neural networks

1. **Convolutional Layers:** These are the first layer in a `CNN`. They perform convolution operations on the input data. The main purpose of these layers is to extract features from the input image. Each filter in the convolutional layer is responsible for detecting a specific feature in the image. For example, in the first convolutional layer, the filters might detect edges, corners, or textures.

2. **Pooling Layers:** After the convolutional layers, pooling layers are used to reduce the spatial size of the representation. This is done to make the network invariant to small transformations and to control overfitting. There are several types of pooling layers, but the most common is max pooling, where the maximum value in each patch of the feature map is selected.

3. **Fully-Connected (FC) Layers:** These are the final layers in a `CNN`. They are used to classify the features extracted by the convolutional and pooling layers. Each neuron in a fully-connected layer is connected to every neuron in the previous layer. The output of the fully-connected layer is a set of outputs, one for each class. The class with the highest output value is the network's prediction for the input.

<div align="center">

![GPT-2](https://www.mdpi.com/entropy/entropy-19-00242/article_deploy/html/images/entropy-19-00242-g001-550.jpg)

</div>

### Function Activation

**Function activation** is a key aspect of `CNNs`. In a `CNN`, each neuron in a layer is connected to every neuron in the previous layer. When a neuron is activated, it sends a signal to the next layer. The activation function decides whether a neuron should be activated or not, based on the stimuli it is receiving from the previous layer.

There are several types of activation functions used in `CNNs`, including:

- **Sigmoid function:** This function maps the input values to the range [0, 1]. It's often used in the output layer of binary classification problems.

- **Tanh function:** This function maps the input values to the range [-1, 1]. It's often used in the hidden layers.

- **ReLU (Rectified Linear Unit) function:** This function is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero. It's computationally efficient and avoids the vanishing gradient problem.

<div style="text-align:center; margin-top: 20px;">
    <img src="https://b2633864.smushcdn.com/2633864/wp-content/uploads/2021/05/relu_activation.png?lossy=2&strip=1&webp=1" alt="CNN">
</div>

## Transfer Learning

### MobileNet v2

The `MobileNet v2` architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input. `MobileNet v2` uses lightweight depthwise convolutions to filter features in the intermediate expansion layer. Additionally, non-linearities in the narrow layers were removed in order to maintain representational power.

<div align="center">

![GPT-2](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-06_at_10.37.14_PM.png)

</div>

### Freezing the layers

**Freezing layers** in a neural network is a technique used to prevent the weights of certain layers from being updated during training. This is often used in transfer learning, where a pre-trained model is used as a starting point on a second task. By freezing the layers of the pre-trained model, the model can be used as a feature extractor, and only the weights of the added layers are learned during fine-tuning.

## Summarize

In conclusion, the evaluation of the two models in the image classification task produced excellent results indicating their high ability to accurately identify and classify visual data. This success highlights the potential of the models for practical applications in various domains where sophisticated image analysis capabilities are required.
