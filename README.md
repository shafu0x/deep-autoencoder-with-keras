# deep-autoencoder-using-keras

## Getting Started
An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore signal “noise.” Along with the reduction side, a reconstructing side is learnt, where the autoencoder tries to generate from the reduced encoding a representation as close as possible to its original input, hence its name.

In this example I use an autoencoder to encode/decode images from the MNIST dataset. In the first step the images are encoded to a smaller dimension (type of dimensionality reduction). After that the task of the decoder is to decode that lower dim. image to its original form.

![images](https://user-images.githubusercontent.com/28685502/43400785-396a18f8-942c-11e8-9251-807f13d5dd1c.png)


## Dataset 
The code downloads the mnnist dataset automatically if it does not exist on your local machine. Alternatively, download the four zip fles and place them in a directory and name it according to the code.

## Results
![untitled](https://user-images.githubusercontent.com/28685502/43400628-d4d02928-942b-11e8-9895-1adcfe62d4d6.png)
