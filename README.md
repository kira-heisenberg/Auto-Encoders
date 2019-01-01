# why autoencoders?

Autoencoders are a very good classifiers when it comes to classifing medical images as datasets are very small there is a significant advantage of using its pretrained layers as it is a unsupervised learning.

## usage
Download the data set at:
https://challenge2018.isic-archive.com/task3/training/

Download MNIST and CIFAR-10

# how far can an auto encoder regenerate images?

For this cifar-10 and MNIST are considered is considered and trained for Epochs = 100,batch size = 250,learning rate= 0.00001,Adams optimizer

### Train autoencoder on MNIST using the command

```bash
python autoencoder.py

```

![alt text](https://github.com/saiky-cheeku/Auto-Encoders/blob/master/ae_out.png)


### Train autoencoder on CIFAR-10 using the command

```bash
python ae_cifar.py
```

![alt text](https://github.com/saiky-cheeku/Auto-Encoders/blob/master/cifar_output.png)


    
