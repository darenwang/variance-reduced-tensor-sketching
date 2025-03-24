import numpy as np
import math


class sphere_density():
    def __init__(self,dim):
        self.dim=dim
        self.density= 1/(1- self.sphere_volume(1/2))
    



    def sphere_volume(self, r):
        return (math.pi ** (self.dim / 2)) * (r ** self.dim) / math.gamma(self.dim / 2 + 1)
    
    def density_value(self,x_input):
        if np.linalg.norm(2*x_input-1)>1:
            return self.density
        return 0
    
    def generate (self, N):
        if self.dim==2:
            factor=10
        else:
            factor=3
        data=np.random.uniform(low=-1, high=1, size=factor*N*self.dim).reshape(factor*N, self.dim)
        row_norms = np.linalg.norm(data, axis=1)
        data=data[row_norms>1][:N]
        return data/2+1/2
    
""" 
distribution= sphere_density(2)
data=distribution.generate(100)
data.shape
distribution.sphere_volume(1)
np.linalg.norm(2*data[0]-1)

x = data[:, 0]
y = data[:, 1]

# Create the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, marker='o', color='blue')
plt.title("Scatter Plot of 2D Data")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show(block=False)
N=100
dim=2
factor=20
data=np.random.uniform(low=-1, high=1, size=factor*N*dim).reshape(factor*N, dim)
row_norms = np.linalg.norm(data, axis=1)
sum(row_norms>1)
data.shape
"""