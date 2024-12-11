import numpy as np

class Numpy_Error_Handler:

    def __init__(self) -> None:
        pass 

    #INFO This is for the underflow error in multiplication due to value being too small 
    def underflow_gradients(self,gradient, min_val=1e-10,max_val = 1e10):
        return np.clip(gradient,min_val,max_val)
    

    #INFO This is for the overflow error in functions 
    def overflow_gradients(self, gradient , min_val=-700 , max_val=700):
        return np.clip(gradient, min_val,max_val)
    
    #INFO Safe Log to prevent dividing by 3
    def safe_log(self, x , epsilon=1e-10):
        return np.log(np.clip(x,epsilon,None))