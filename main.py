import numpy as np

from gaussian_mixture import  gaussian_mixture


from kde import kernel_density

from tensor_estimate  import vrs_prediction
  
#from full_tensor import compute_full_tensor
 

dim=7
N_train= 10000

##############tuning parameter selection
MM=10
if N_train<2**dim*MM:
    print('insufficient data')
    LL =1
else:
    LL = 2

print(MM,LL)
tensor_shape=[LL for _ in range(dim)]

tensor_shape[0]=MM

#########################################

lr_rec=0
kde_rec=0



distribution=gaussian_mixture(dim, [1,-1], [1,0.5])
for rr in range(10):
    
    
    X_train= distribution.generate(N_train)
    N_test=10000
    X_test= distribution.generate(N_test)
    
    #############density transform 
    vrs_model=vrs_prediction(tensor_shape, dim, MM, X_train)
    y_lr= vrs_model.predict(X_test)
    y_true = np.array([distribution.density_value(xx) for xx in X_test])
    lr_rec += np.linalg.norm(y_lr - y_true,2)**2/np.linalg.norm(y_true,2)**2
    print('lr transform error = ', lr_rec/(rr+1))

     
    
    
    
    y_kde=kernel_density().compute(dim, X_train, X_test)
    err_kde = np.linalg.norm(y_kde - y_true,2)**2/np.linalg.norm(y_true,2)**2

    kde_rec+=err_kde

    print('kde error = ', kde_rec/(rr+1))
