import numpy as np
import traceback
from Error_Reaper import Numpy_Error_Handler as neh
class Discriminator:



    def __init__(self, n_odes , learning_rate , weights = [] , bias = None):
        np.seterr(all='raise')
        self.handler = neh()
        self.nNodes = n_odes 
        # self.weights = np.array(np.random.randn(self.nNodes*self.nNodes))
        # print(f"THIS IS WEIGHTS ==>> {self.weights.shape}")

        if(len(weights) > 0 or bias != None):
            self.weights = weights
            self.bias = bias
        else:
            self.weights = np.array(np.random.randn(self.nNodes , self.nNodes , 3))
            self.bias = np.random.normal()
            # self.bias = np.array(np.random.randn(self.nNodes , self.nNodes , 3))
        self.learningRate = learning_rate
    #INFO Activation Function 
    def sigmoid(self,x):
        try:
            return 1.0 / (1.0 + np.exp(self.handler.underflow_gradients(-x)))
        except FloatingPointError:
            x = self.handler.overflow_gradients(-x)
            return 1.0 / (1.0 + np.exp(x))

    
    def forward(self, x):
        # Forward pass
        results = np.empty((32,32,3))
        
        for i in range(3):
            if(type(self.bias) == np.ndarray and type(x) == np.ndarray):
                results[:,:,i] = self.sigmoid(np.dot(x[:,:,i],self.weights[:,:,i]) + self.bias[:,:,i])
            elif(type(self.bias) == np.ndarray):
                results[:,:,i] = self.sigmoid(np.dot(x,self.weights[:,:,i]) + self.bias[:,: , i])
            elif(type(x) == np.ndarray):
                results[:,:,i] = self.sigmoid(np.dot(x[:,:,i],self.weights[:,:,i]) + self.bias)
            else:
                results[:,:,i] = self.sigmoid(np.dot(x,self.weights[:,:,i]) + self.bias)

        
        # results = results.reshape(32,32,3)
        return results

    
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
        try:
            results = self.handler.safe_log(1-prediction)
            return results 
        except FloatingPointError:
            print(f"This is prediction ==>> {prediction.size} and type ==>> {type(prediction)} and size ==>> {prediction.shape}")
            print(traceback.print_exc())
            exit()    
    def derivatives_from_noise(self, noise):
        prediction = self.forward(noise)
        derivatives_weights = self.handler.underflow_gradients(noise) * self.handler.underflow_gradients(prediction)
        derivative_bias = prediction
        return derivatives_weights, derivative_bias
    
    def update_from_noise(self, noise):
        ders = self.derivatives_from_noise(noise)
        ders = (self.handler.underflow_gradients(gradient=ders[0]), self.handler.underflow_gradients(gradient=ders[1]))

        self.weights -= self.learningRate * ders[0] 
        self.bias -= self.learningRate * ders[1]