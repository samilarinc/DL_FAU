# -*- coding: utf-8 -*-
"""
Created on Sun May 15 13:40:15 2022

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt

class Checker:
    
    def __init__(self, resolution, tile_size):
        self.tile_size = tile_size
        self.resolution = resolution
        # self.output = None
        
    def draw(self):
        amount = int(self.tile_size*2)
        a = np.zeros((amount,amount), dtype = int)
        a[self.tile_size:,:self.tile_size] = 1
        a[:self.tile_size,self.tile_size:] = 1
        num = int(self.resolution/(self.tile_size*2))
        b = np.tile(a,(num,num))
        self.output = b.copy()
        return b
           
    def show(self):
        output = self.draw()
        plt.imshow(output, cmap = "gray")


class Circle:
    
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        # self.output = None
        
    def draw(self):
        res = self.resolution
        x = np.linspace(0,res-1,res).reshape(1,res)
        y = np.linspace(0,res-1,res).reshape(res,1)
        distance = np.sqrt((x - self.position[0])**2 + (y-self.position[1])**2)
        
        circ = distance <= self.radius
        self.output = circ.copy()
        return circ
    
    def show(self):
        output = self.draw()
        plt.imshow(output, cmap = "gray")


class Spectrum:
    
    def __init__(self, resolution):
        self.resolution = resolution
        # self.output = None
        
    def draw(self):
        res = self.resolution
        spectrum = np.zeros([res,res, 3])
        spectrum[:,:,0] = np.linspace(0,1,res)
        spectrum[:,:,1] = np.linspace(0,1,res).reshape(res,1)     
        spectrum[:,:,2] = np.linspace(1,0,res)

        self.output = spectrum.copy()
        return spectrum
    
    def show(self):
        output = self.draw()
        plt.imshow(output)
        
# spectrum[:,:,1] = np.linspace(0,1,res).reshape(res,1)     
# spectrum[:,:,2] = np.linspace(1,0,res)
# tile_size_1 = int(input("tile_size for Checker: "))
# resolution_1 = int(input("resolution for Checker: "))
# Q12 = Checker(tile_size_1,resolution_1)

# if resolution_1 % (tile_size_1*2) != 0:
#     print("Invalid values entered, try again")
# else:  
#     Q12.show()
  
# resolution_2 = int(input("resolution for Circle: "))
# radius = int(input("radius for Circle: "))
# position = input("position for Circle, enter as x,y: ")

# pos = tuple(map(int, position.split(',')))

# Q13 = Circle(resolution_2, radius, pos)
# Q13.show()
  
# resolution_3 = int(input("resolution for Spectrum: "))
# Q14 = Spectrum(resolution_3)
# Q14.show()

