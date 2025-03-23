import numpy as np
from kde import kernel_density

from tensor_prediction_stage import vrs_prediction
from generate import general_normal_density
 
 



dim=5
N_train=5000



##############tuning parameter selection
MM=5
if N_train<2**dim*MM:
    print('insufficient data')
    LL =1
else:
    LL=2
#LL=2
tensor_shape=[LL for _ in range(dim)]
tensor_shape[0]=MM
print('M=', MM, 'L=',  LL)
#########################################

lr_error=0
kde_error=0

means_1=[dd/dim for dd in range(dim)]
means_2= [-1*dd/dim for dd in range(dim)]
distribution=general_normal_density(dim,means_1, np.ones(dim)*0.3, means_2, np.ones(dim)*0.4 )
 
 



for rr in range(10):
    
 
    X_train= distribution.generate(N_train)
    vrs = vrs_prediction( tensor_shape, dim, MM , X_train)

    N_test = 10000
    X_new= np.random.uniform(0,1, N_test*dim).reshape((N_test, dim))

    y_lr=np.array([vrs.predict(xx) for xx in X_new])
    
    
    y_true = np.array([distribution.density_value(xx) for xx in X_new])
    
    lr_error+=np.linalg.norm(y_lr - y_true,2)**2/np.linalg.norm(y_true,2)**2
    
    y_kde=kernel_density().compute(dim, X_train, X_new)
    kde_error+=np.linalg.norm(y_kde - y_true,2)**2/np.linalg.norm(y_true,2)**2
    print("lr errors =",  lr_error/(rr+1), "kde errors =", kde_error/(rr+1))



