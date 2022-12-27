# COSE474

## Project #1: MLP Implementation

### Implement 2-Layer Neural Net with Sofmax Classifier

• Perform the image classification using “CIFAR-10” dataset.

• Two weights W1, W2 with biased b1, b2.

• Predicted output y' = W2(relu(W1x + b1) + b2.

• Total loss = data loss (softmax+log likelihood loss) + L-2 regularization loss (to W1, W2, not b1, b2).

• The Ipython Notebook “two_layer_net.ipynb” will walk you through the implementation of a two-layer neural network classifier.

## Project #2: CNN Architecture Implementation

### Implement “ResNet-50” 

• Train “ResNet-50” model with “CIFAR-10” datasets 

• Optimize parameters with Adam optimizer and cross Entropy Loss 
  
  ■ Get “CIFAR-10” dataset with torchvision library
  
• Procedure 

    1) Load the trained model (which is given)
    2) Complete the class ResNet50_layer4 in “resnet50_skeleton.py”.
    3) Train it with CPU or GPU and submit the screen capture of test accuracy as a result.
    4) You can use a trained checkpoint parameters of 285 epochs.
       You will train model only 1 epoch.
       
#### Question 1: Implement the “bottleneck building block”

• Bottleneck building block (residual block) 

• For each residual function F, we use a stack of 3 layers. The three layers are 1x1, 3x3 and 1x1 convolutions.

• Input: 𝑥 

• Output: relu(𝐹(𝑥) + 𝑥) 

#### Question 2: Implement the “ResNet50_layer4” 

• Network Architecture 

• It is different from the original ResNet50. 

• Complete the network code at the table. 

## Project #3: Encoder-Decoder Implementation

### Train “UNet”
– Network Architecture Contracting path Channel size Expanding path

Compared to the skeleton code, image size changed because we are using pascal VOC dataset

    1) U-net architecture (example for 32x32 pixels in the lowest resolution)
    2) Each blue box corresponds to a multi-channel feature map.
    3) The number of channels is denoted on top of the box.
    4) The x-y-size is provided at the lower left edge of the box.
    5) White boxes represent copied feature maps.
    6) The arrows denote the different operations.

• It consists of a contracting path (left side) and an expansive path (right side).

• The contracting path follows the typical architecture of a convolutional network.

• It consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling.

• At each downsampling step we double the number of feature channels.

• Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution (up-convolution) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU.

• The cropping is necessary due to the loss of border pixels in every convolution.

• At the final layer a 1x1 convolution is used to map each 64- component feature vector to the desired number of classes.

• In total the network has 23 convolutional layers.

### Implement “ResNet-encoder-Unet” Layer number
– Replace the encoder of original U-net with ResNet-50
