#from polynomial import polynomial_new_basis
from wavelet import wavelet_new_basis

import numpy as np

from tensor_training_stage import tensor


 
class generate_estimated_basis_vector:
    def __init__(self, dim,M ):
        self.basis=wavelet_new_basis(M)
        self.dim=dim

    #basis_val[d][m] is the m-th basis evaluated at x[d]
    def compute_basis_val(self, P_x_basis,x):
        
        basis_val=[self.basis.compute_single_x_all_new_basis ( P_x_basis[d],x[d]) for d in range(self.dim)]

        return basis_val


#generate_new_basis_vector(dim, P_x_basis,MM).compute_basis_val(X_train[0])
######################




class tensor_prediction:
    def __init__(self, tensor_shape, dim,M,X_train):
        self.P_x_basis=[]
        self.dim=dim 
        self.new_shape=[]
        self.generate_new_basis_vector=generate_estimated_basis_vector(dim,M)
        for dd in range(dim):
            print('coordinate =',dd)
            
            self.P_x_basis.append( tensor(dim,dd).compute_range_basis(X_train,tensor_shape) ) 
            self.new_shape.append(len(self.P_x_basis[-1]))
        
         
        #print(self.P_x_basis, self.new_shape)
        self.A_predict=np.zeros(self.new_shape, dtype=float)
        for i in range(len(X_train)):
            
            self.A_predict=self.A_predict+self.compute_rank_one(X_train[i])        
            
        self.A_predict=self.A_predict/len(X_train)
        #print(self.A_predict)
        
    def compute_rank_one(self,x_input):
        basis_val=self.generate_new_basis_vector.compute_basis_val( self.P_x_basis,x_input)


        #we compute rank 1 tensor given basis_val
        #the rank one tensor is 
        #[p_0(x[0]), p_1(x[0]) , ...  ]\otimes [p_0(x[1]),p_1(x[1]),... ]\otimes [p_0(x[d-1]), p_1(x[d-1]),... ]
        temp_tensor=1
        
        for dd in range(self.dim-1, -1, -1):
            temp_tensor= np.tensordot( np.reshape(basis_val[dd], (len(basis_val[dd]),1)), [temp_tensor],axes=(1,0))

        
        return temp_tensor 
    
    def predict(self,x_test):
        basis_val=self.generate_new_basis_vector.compute_basis_val( self.P_x_basis,x_test)
        ans= np.tensordot(basis_val[0],self.A_predict,axes=(0,0) )
        for dd in range(1,self.dim):
            ans= np.tensordot(basis_val[dd],ans,axes=(0,0) )
        return ans
  
    
  
    
############### transform domain
class domain:
    def __init__(self,dim, factor,X_train):
        self.X_train=X_train
        self.dim=dim
        Y=X_train.transpose()
        self.upper=[]
        self.lower=[]
        for dd in range(dim):
            self.upper.append(np.quantile(Y[dd],1-factor))
            self.lower.append( np.quantile(Y[dd],factor))
        self.upper=np.array(self.upper)+1e-07
        self.lower=np.array(self.lower)-1e-07
        #print(self.upper)
        #print(self.lower)
        self.difference =self.upper-self.lower
        self.density_factor=np.prod(self.difference)
        
        
    def transform_to_0_1(self, x_vec ):
        
        #slope=1/(upper-lower)
        #intercept=-1*lower*slope
        return [( x_vec[dd]-self.lower [dd])/(self.difference[dd]) for dd in range(self.dim)]

    #def transform_to_orignial(self, y_vec):
        #y*(upper-lower )+lower
    #   return  [y_vec[dd]*self.difference[dd] +self.lower[dd]  for dd in range(self.dim)]

    def compute_data(self,XX):
        X_transform=[self.transform_to_0_1(XX[ii]) for ii in range(len(XX)) ]
        return np.array(X_transform)
    
    def transform_density_val(self, val):
        return val/self.density_factor
#new=domain(dim, 0.005, X_train)
#X_transform=new.compute_data()



##########
class vrs:
    def __init__(self,tensor_shape, dim, MM, X_train,domain_factor=0.01):
        self.new_domain=domain(dim, domain_factor, X_train)
        X_train_transform=self.new_domain.compute_data(X_train)
        self.lr_transform= tensor_prediction( tensor_shape, dim, MM , X_train_transform)
    
    def predict(self,X_test):
        X_test_transform= self.new_domain.compute_data(X_test)
        
        
        return np.array([self.new_domain.transform_density_val(self.lr_transform.predict(xx)) for xx in X_test_transform])
            
  
    
  