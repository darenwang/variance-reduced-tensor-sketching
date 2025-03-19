import numpy as np



from scipy.special import legendre



#legendre polynomial on [0,1]
class polynomial:
    def __init__(self,max_degree=20) -> None:
        self.max_degree=max_degree
        self.coefficients=[]
        for i in range(max_degree+1):
            self.coefficients.append(np.array(legendre(i))* np.sqrt(2*i+1))
 

    def compute_single_x_single_degree(self, degree,powers_x):
        return np.dot( self.coefficients[degree], powers_x[-degree-1:]) 
    
    def compute_single_x_all_degree(self,  cur_shape,x):
        powers_x= np.ones(cur_shape)
        new_x=2*x-1
        for i in range(cur_shape-2, -1,-1):
            powers_x[i]=powers_x[i+1]*new_x
        
        return np.array([ self.compute_single_x_single_degree(degree, powers_x) for degree  in range(cur_shape)  ])

 

class polynomial_new_basis:
    def __init__(self,M) -> None:
        self.M=M
        self.coefficients=[]
        
        for i in range(M):
            self.coefficients.append(np.concatenate ((np.zeros(M-i-1) , np.array(legendre(i))* np.sqrt(2*i+1) )))
        
        
        #print(self.new_coefficients.shape)
        #print(np.dot(self.new_coefficients[0],[1,1,1]))
    def compute_single_x_all_new_basis(self ,U,x):
        self.new_coefficients= np.matmul(U,self.coefficients)
        self.new_length=len(U)
        powers_x= np.ones(self.M)
        new_x=2*x-1
        for i in range(self.M-2, -1,-1):
            powers_x[i]=powers_x[i+1]*new_x
        
        return np.array([  np.dot(self.new_coefficients[index], powers_x) for index  in range(self.new_length)  ])


 