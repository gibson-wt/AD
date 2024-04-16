import numpy as np
import roughpy as rp
from Visualise import get_increments
from dataclasses import dataclass
from numpy import ndarray

class MinMaxAugmentation:
    def __init__(self,stream,resolution=14):
        self.stream = stream
        self.width = stream.width
        self.resolution=resolution

    def augment(self,signature,interval,depth):
        increments=get_increments(self.stream,interval,self.resolution)
        path = np.cumsum(increments,axis=0)
        minimum = path.min(0)
        maximum = path.max(0)

        context = rp.get_context(self.width,depth,rp.DPReal)

        j=0
        for basis in context.tensor_basis:
            if j==0:
                j+=1
                continue 
            string = str(basis)
            contexts = string.split('(')[1].split(')')[0].split(',')
            int_context  = np.array([int(x) for x in contexts])
            scalings = np.ones((self.width,))
            for d in range(1,self.width+1):
                scalings[d-1] = (maximum[d-1]-minimum[d-1])**((int_context==d).sum())
            signature[j] = signature[j] / np.prod(scalings)
            j+=1
        return signature

    def augment_multiple(self,signatures, intervals, depth):
        assert signatures.shape[1] == (self.width ** (depth+1) - 1)/(self.width-1)
        assert signatures.shape[0] == len(intervals)

        for i in range(signatures.shape[0]):
            signatures[i] = self.augment(signatures[i],intervals[i],depth)
            
        return signatures
    
class ReflectAugmentation:

    def __init__(self,width):
        self.width = width

    def augment(self,signature,depth,dim):

        context = rp.get_context(self.width,depth,rp.DPReal)

        j=0
        for basis in context.tensor_basis:
            if j==0:
                j+=1
                continue 
            string = str(basis)
            contexts = string.split('(')[1].split(')')[0].split(',')
            int_context  = np.array([int(x) for x in contexts])
            scalings = (-1)**((int_context==dim).sum())
            signature[j] = signature[j] * scalings
            j+=1
        return signature
    
# from sigwgan code
@dataclass
class AddTime:

    def apply(x: ndarray):
        t = np.linspace(0,1,x.shape[0]).reshape(-1,1)
        return np.concatenate([t,x],axis=1)
    
@dataclass
class CumSum:
    def apply(x: ndarray):
        x = x.reshape(-1,1)
        cs = np.cumsum(x).reshape(-1,1)
        return np.concatenate([x,cs],axis=1)