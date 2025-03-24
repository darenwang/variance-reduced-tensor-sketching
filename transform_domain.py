import numpy as np



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
