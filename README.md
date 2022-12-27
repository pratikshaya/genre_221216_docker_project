# genre_221216_docker_project

The pre-trained model and its block diagram can be found in the link https://arxiv.org/pdf/1703.09179.pdf

Exponential linear unit (ELU) is used as an activation function in all convolutional layers. Max-pooling of (2, 4), (4, 4), (4, 5), (2, 4), (4, 4) is applied after every convolutional layer respectively. In all the convolutional layers, the kernel sizes are (3, 3), numbers of channels N is 32, and Batch normalisation is used. The input has a single channel, 96-mel bins, and 1360 temporal frames. After training, the feature maps from 1st– 4th layers are subsampled using average pooling while the feature map of 5th layer is used as it is, since it is already scalar (size 1 × 1). Those 32-dimensional features are concatenated to form a convnet feature.
![image](https://user-images.githubusercontent.com/26374302/209610143-798e83fc-f2c3-4e6b-b70d-8d5038113a7a.png)
