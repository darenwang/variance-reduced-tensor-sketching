import numpy as np
from generate import general_normal_density 

####kernel density estimation
from kde import kernel_density
########VRS
from tensor_prediction_stage import VRS_prediction

########


 


#####################   Use  mixture of Gasussian as the underlining distribution
dim=8
N_train= 5000

means=[dd/dim for dd in range(dim)]
distribution=general_normal_density(dim,means, np.ones(dim)*0.3, np.ones(dim)*0.9, np.ones(dim)*0.3 )

##############tuning parameter selection
MM=20
if N_train<2**dim*MM:
    print('insufficient data')
    LL =1
else:
    LL=2

print(MM,LL)
tensor_shape=[LL for _ in range(dim)]
tensor_shape[0]=MM

#########################################
####errors 
vrs_err=0
kde_err=0

########

 

for rr in range(30):
    
     
    #generate training and test data
    X_train= distribution.generate( N_train)
    N_test = 10000
    #the test data are used to compute the L_2 errors of density estimators
    X_test= np.random.uniform(0,1, N_test*dim).reshape((N_test, dim))

    
    ######VRS estimation
    VRS= VRS_prediction( tensor_shape, dim, MM , X_train)
    y_vrs=np.array([VRS.predict(xx) for xx in X_test])
    ##############
    
    ###############kde
    y_kde=kernel_density().compute(dim, X_train, X_test)
    #############################
    
    #############find the true density at test data for computing error    
    y_true = np.array([distribution.density_value(xx) for xx in X_test])
    ########################
    
    ################compute errors
    vrs_err += np.linalg.norm(y_vrs - y_true,2)**2/np.linalg.norm(y_true,2)**2
    
    kde_err  +=np.linalg.norm(y_kde - y_true,2)**2/np.linalg.norm(y_true,2)**2

    print('vrs average error = ', vrs_err/(rr+1))
    print('kde average error = ', kde_err/(rr+1))


