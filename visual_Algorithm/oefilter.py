import math
import time
import numpy as np
class oefilter:
    def __init__(self,  mincutoff, beta, dcutoff):   
        self.x = None
        self.y = None
        self.z = None
        
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
        
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
    def __call__(self, x, y, z, te):
        if self.x is None:
           self.x = x
           self.y = y
           self.z = z
           return x, y, z
        edx = (x - self.x) / te
        edy = (y - self.y) / te
        edz = (z - self.z) / te
        dalpha = self._alpha(self.dcutoff, te)
        self.dx = self.dx + (dalpha * (edx - self.dx))
        self.dy = self.dy + (dalpha * (edy - self.dy))
        self.dz = self.dz + (dalpha * (edz - self.dz))
        speed = np.sqrt(self.dx**2 + self.dy**2 + self.dz**2)
        cutoff = self.mincutoff + self.beta * speed
        alpha = self._alpha(cutoff, te)
        res_x = self.x + alpha * (x - self.x)
        res_y = self.y + alpha * (y - self.y)
        res_z = self.z + alpha * (z - self.z)
        self.x, self.y, self.z = res_x, res_y, res_z
        return res_x, res_y, res_z
    def _alpha(self, cutoff, te):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)
    
        
        