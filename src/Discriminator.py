import numpy as np
class Discriminator:



    def __init__(self, n_odes , learning_rate , weights = [] , bias = None):
        self.nNodes = n_odes 
        # self.weights = np.array(np.random.randn(self.nNodes*self.nNodes))
        # print(f"THIS IS WEIGHTS ==>> {self.weights.shape}")

        if(len(weights) > 0 or bias != None):
            self.weights = weights
            self.bias = bias
        else:
            self.weights = np.array(np.random.randn(self.nNodes))
            self.bias = np.random.normal()

        self.learningRate = learning_rate

    def sigmoid(self,x):
        # print(f"THis is the fail state x ==> {x}")
        # a = np.exp(x)
        # b = (1.0+np.exp(x))
        
        # print(f"This is b ==> {b}")
        
        # result = a/b

        return 1.0 / (1.0 + np.exp(-x))

    
    def forward(self, x):
        # Forward pass
        dot = np.dot(x, self.weights) 
        # print(f"THIS IS DOT {dot}")
        
        return self.sigmoid(np.dot(x, self.weights) + self.bias)


    
    def error_from_image(self, image):
        prediction = self.forward(image)
        # We want the prediction to be 1, so the error is -log(prediction)
        try:
            return -np.log(prediction)
        except:
            print("This is preditcion " + str(prediction))
            exit()
    
    def derivatives_from_image(self, image):
        prediction = self.forward(image)
        
        derivatives_weights = -image * (1-prediction)
        derivative_bias = -(1-prediction)
        return derivatives_weights, derivative_bias
    
    def update_from_image(self, x):
        ders = self.derivatives_from_image(x)
        self.weights -= self.learningRate * ders[0]
        self.bias -= self.learningRate * ders[1]

    def error_from_noise(self, noise):
        prediction = self.forward(noise)
        # We want the prediction to be 0, so the error is -log(1-prediction)
        return -np.log(1-prediction)
    
    def derivatives_from_noise(self, noise):
        prediction = self.forward(noise)
        derivatives_weights = noise * prediction
        derivative_bias = prediction
        return derivatives_weights, derivative_bias
    
    def update_from_noise(self, noise):
        ders = self.derivatives_from_noise(noise)
        self.weights -= self.learningRate * ders[0]
        self.bias -= self.learningRate * ders[1]