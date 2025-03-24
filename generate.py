import numpy as np


from scipy.stats import truncnorm
from scipy.stats import bernoulli


#####two mixture normal 

class normal_density():
    def __init__(self,dim, trunc_mean,trunc_dev):
        self.dim=dim
        self.sample_model_1= truncnorm(loc=trunc_mean[0],scale=trunc_dev[0], a=-1*trunc_mean[0]/trunc_dev[0], b=( (1  - trunc_mean[0])/trunc_dev[0]))

        self.sample_model_2= truncnorm(loc=trunc_mean[1],scale=trunc_dev[1], a=-1*trunc_mean[1]/trunc_dev[1], b=( (1  - trunc_mean[1])/trunc_dev[1]))
        self.prob=0.5
    
    def density_value(self,x):

        return self.prob*np.prod([ self.sample_model_1.pdf(xx) for xx in x ])+ (1-self.prob)*np.prod([ self.sample_model_2.pdf(xx) for xx in x ])
        

    def generate(self,N):
        

        rr= np.tensordot([bernoulli.rvs(self.prob, size=N)] , [np.ones(self.dim)]  ,axes=[[0],[0]]  )
        #data1= np.multiply(rr,np.random.uniform(0,1, N*self.dim).reshape((N, self.dim)))
        data1 = np.multiply( rr,self.sample_model_1.rvs(size=N* self.dim).reshape((N, self.dim)))

        data2 = np.multiply( -1*rr+1,self.sample_model_2.rvs(size=N* self.dim).reshape((N, self.dim)))
        
        return data1+data2
    
##################    
    
class general_normal_density():
    def __init__(self,dim, trunc_mean_1,trunc_dev_1, trunc_mean_2,trunc_dev_2 ):
        self.dim=dim
        self.sample_model_1=[]
        self.sample_model_2=[]
        for dd in range(dim):
            self.sample_model_1.append( truncnorm(loc=trunc_mean_1[dd],scale=trunc_dev_1[dd], a=-1*trunc_mean_1[dd]/trunc_dev_1[dd], b=( (1  - trunc_mean_1[dd])/trunc_dev_1[dd])))
            self.sample_model_2.append(truncnorm(loc=trunc_mean_2[dd],scale=trunc_dev_2[dd], a=-1*trunc_mean_2[dd]/trunc_dev_2[dd], b=( (1  - trunc_mean_2[dd])/trunc_dev_2[dd])))

        self.prob=0.5
    
    def density_value(self,x):
        ans_1= np.prod([ self.sample_model_1[dd].pdf(x[dd]) for dd in range(self.dim) ])
        ans_2= np.prod([ self.sample_model_2[dd].pdf(x[dd]) for dd in range(self.dim) ])
        return self.prob*ans_1+ (1-self.prob)*ans_2
        

    def generate(self,N):
        data1=[]
        data2=[]
        for dd in range(self.dim):
            data1.append( self.sample_model_1[dd].rvs(size=N))
            data2.append( self.sample_model_2[dd].rvs(size=N))
        data1=np.array(data1)
        data2=np.array(data2)
        data1=data1.transpose()
        data2= data2.transpose()
        #print(data1)
        rr= np.tensordot([bernoulli.rvs(self.prob, size=N)] , [np.ones(self.dim)]  ,axes=[[0],[0]]  )
        data1 = np.multiply( rr, data1)

        data2 = np.multiply( -1*rr+1,data2)
        
        return data1+data2    
    
    
   