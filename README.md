# DigitRecognizer

### Fully Connected Neural Network
I created two simple versions of a digit recognizer model using Pytorch and the MNIST dataset in order to learn basic ML and Computer Vision Concepts.
The first model uses a flatten layer to flatten the greyscale images into a single dimension and then uses a few Linear layers with ReLU layers in between to introduce non-Linearity (These are needed to find certain features such as the curves in the digits). My architecture is essentially similar to one seen in the Chapter 1 deep learning video from 3Blue1Brown, which is actually what inspired me to do this mini-project. 

Here is a link to the video: [But what is a neural network? | Chapter 1, Deep learning](https://www.youtube.com/watch?v=aircAruvnKk&t=11s)

### Convolutional Neural Network
The second model uses a Convolutional Neural Network(CNN) architecture as opposed to the Fully Connected Neural Network architecture used in the first model. CNN's are commonly used for Image Recognition or Image Classification tasks as they excel at detecting local features and patterns in images. For my second model, I followed a similar architecture to the one used in the CNN Explorer website. 

Here is a link to this website: [CNN Explorer](https://poloclub.github.io/cnn-explainer/)

### Clarifications
It's worth noting that a Fully Connected Neural Network architecture works here due to the simplicity of the task and the well structured nature of the dataset. I somewhat highlight this when I point out in the CNN notebook that the CNN model took longer to go through the epochs than the base model(Fully Connected Neural Network Model). This is because the task is relatively simple and the CNN model has more layers and complexity too it. Given a more complex task, such as classifying RGB images of different types of animals in nature, a Fully Connected Neural Network would be unfeasible given that the number of parameters would grow exponentially which would be too computationally expensive. As mentioned before, CNN's would perform better for these kind of tasks due to the nature of Convolutional layers and the use of Pooling layers, which helps reduce the computational overhead. As one would expect, the second model(CNN Model) would take less time to go through all the epochs than the Fully Connected Neural Network model given the more complex classification task.

Finally, I also wanted to link this website that I used to learn the basics of Pytorch: [learnpytorch.io](https://www.learnpytorch.io)

This website was created by Daniel Bourke and is a part of a free comprehensive course that goes over the fundamentals of Pytorch
