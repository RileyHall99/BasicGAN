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
            print("no DATA")
            self.weights = np.array(np.random.randn(self.nNodes , self.nNodes , 3))
            self.bias = np.random.normal()
            # self.bias = np.array(np.random.randn(self.nNodes , self.nNodes , 3))

        self.learningRate = learning_rate
    #INFO Activation Function 
    def sigmoid(self,x):
        # print(f"THis is the fail state x ==> {x}")
        # a = np.exp(x)
        # b = (1.0+np.exp(x))
        
        # print(f"This is b ==> {b}")
        
        # result = a/b
        try:
            return 1.0 / (1.0 + np.exp(-self.clip_gradients(x)))
        except FloatingPointError:
            x = np.clip(x,-700,700)
            return 1.0 / (1.0 + np.exp(-x))

    
    def forward(self, x):
        # Forward pass
        results = np.empty((32,32,3))
        # print("This is bias" + str(type(self.bias)))
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

    def clip_gradients(self,gradient, min_val=1e-10,max_val = 1e10):
        # print(f"This is type of gradient {type(gradient)}")
        return np.clip(gradient,min_val,max_val)
    
    def error_from_image(self, image):
        prediction = self.forward(image)
        # We want the prediction to be 1, so the error is -log(prediction)
        try:
            return -np.log(prediction)
        except:
            print("This is preditcion " + str(prediction))
            exit()
    
    def derivatives_from_image(self, image):
        # print(f"This is x/image ==> {image} and type {type(image)}")
        prediction = self.forward(image)
        # print(f"This is the predition {prediction} the type {type(prediction)}")
        # exit()
        derivatives_weights = -image * (1-prediction)
        derivative_bias = -(1-prediction)
        return derivatives_weights, derivative_bias
    
    def update_from_image(self, x):
        ders = self.derivatives_from_image(x)
        self.weights -= self.learningRate * ders[0]
        self.bias -= self.learningRate * ders[1]

    def safe_log(self, x , epsilon=1e-10):
        return np.log(np.clip(x,epsilon,None))

    def error_from_noise(self, noise):
        
        prediction = self.forward(noise)
        # We want the prediction to be 0, so the error is -log(1-prediction)
        try:
            results = self.safe_log(1-prediction)
            return results 
        except FloatingPointError:
            print(f"This is prediction ==>> {prediction.size} and type ==>> {type(prediction)} and size ==>> {prediction.shape}")
            print(traceback.print_exc())
            exit()    
    def derivatives_from_noise(self, noise):
        prediction = self.forward(noise)
        derivatives_weights = self.clip_gradients(noise) * self.clip_gradients(prediction)
        derivative_bias = prediction
        return derivatives_weights, derivative_bias
    
    def update_from_noise(self, noise):
        ders = self.derivatives_from_noise(noise)
        # print(f"This is type of ders {type(ders)} and at 0 {type(ders[0])}")
        ders = (self.clip_gradients(gradient=ders[0]), self.clip_gradients(gradient=ders[1]))
        self.weights -= self.learningRate * ders[0]
        self.bias -= self.learningRate * ders[1]