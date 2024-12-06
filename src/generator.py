import numpy as np
import matplotlib.pyplot as plt
from pixel import Pixel 
class Generator:
    def __init__(self , n_nodes , learning_rate , weights = [] , bias = []):
        self.nNodes = n_nodes #INFO This is equal to how many nodes per layer 
        # self.weights = np.array(np.random.randn(self.nNodes, 4, self.nNodes)) #INFO This is equal to the amount of nodes*3 since there are going to be 3 layers 
        # self.biases = np.array(np.random.randn(self.nNodes, 4,self.nNodes ))#INFO This is equal to the amount of nodes*3 since there are goingt to be 3 layers 
        if(len(weights) > 0 or len(bias) > 0):
            self.weights = weights
            self.biases = bias
        else:
            self.weights = np.array(np.random.randn(self.nNodes))
            self.biases = np.array(np.random.randn(self.nNodes))

        self.z = 0
        self.learningRate = learning_rate

    #INFO This will ran at the start to begin the machine learning process 
    def generate_image(self)->list:
        image = []
        for i in range(self.nNodes):
            image.append([])
            for j in range(self.nNodes):
                image[i].append([np.random.randint(256) , np.random.randint(256) , np.random.randint(256)])
        # print(image)
        return image 
    
    def sigmoid(self,x):
        # return np.exp(x)/(1.0+np.exp(x))
        # print(f"THis is x ==>> {x}")
        # exit()
        return 1.0 / (1.0 + np.exp(-x))
 
    
    def forward(self, x):
        return self.sigmoid(self.z * self.weights + self.biases)
    
    def error(self, z, discriminator):
        x = self.forward(z)
        # We want the prediction to be 0, so the error is -log(1-prediction)
        y = discriminator.forward(x)
        return -np.log(y)

    def derivatives(self, z, discriminator):
        discriminator_weights = discriminator.weights
        discriminator_bias = discriminator.bias
        x = self.forward(z)
        y = discriminator.forward(x)
        factor = -(1-y) * discriminator_weights * x *(1-x)
        derivatives_weights = factor * z
        derivative_bias = factor
        return derivatives_weights, derivative_bias

    def update(self, z, discriminator):
        error_before = self.error(z, discriminator)
        ders = self.derivatives(z, discriminator)
        self.weights -= self.learningRate * ders[0]
        self.biases -= self.learningRate * ders[1]
        error_after = self.error(z, discriminator)

    def display_image(self, image):
        plt.imshow(image)
        plt.show()

    
# G = Generator(100 , 1.1)

# image = G.generate_image()

# G.display_image(image)