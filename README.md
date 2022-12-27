# genre_221216_docker_project

The pre-trained model and its block diagram can be found in the link https://arxiv.org/pdf/1703.09179.pdf

Exponential linear unit (ELU) is used as an activation function in all convolutional layers. Max-pooling of (2, 4), (4, 4), (4, 5), (2, 4), (4, 4) is applied after every convolutional layer respectively. In all the convolutional layers, the kernel sizes are (3, 3), numbers of channels N is 32, and Batch normalisation is used. The input has a single channel, 96-mel bins, and 1360 temporal frames. After training, the feature maps from 1st– 4th layers are subsampled using average pooling while the feature map of 5th layer is used as it is, since it is already scalar (size 1 × 1). Those 32-dimensional features are concatenated to form a convnet feature.

The parameters of the network can be shown below

![image](https://user-images.githubusercontent.com/26374302/209610390-8c786416-96b6-4a4f-b35d-a86e4c69c351.png)

The architecture for fine tuning the five classes of genre is shown below:

![image](https://user-images.githubusercontent.com/26374302/209610462-ec579891-4122-412f-aa41-660d8ddfe49e.png)
 
 

