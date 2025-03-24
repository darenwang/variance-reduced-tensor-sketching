import numpy as np

from kde import kernel_density
from generate import general_normal_density
from sphere import sphere_density

from tensor_prediction_stage import tensor_prediction
  
#from full_tensor import compute_full_tensor
 

dim=8
N_train= 10000

##############tuning parameter selection
MM=20
if N_train<2**dim*MM:
    print('insufficient data')
    LL =1
else:
    LL=2

print(MM,LL)
#########################################

lr_rec=0
kde_rec=0


means=[dd/dim for dd in range(dim)]

distribution=general_normal_density(dim,means, np.ones(dim)*0.3, np.ones(dim)*0.5, np.ones(dim)*0.5 )
 
#distribution=sphere_density (dim)


for rr in range(10):
    
    nn=N_train 
    X_train= distribution.generate(nn)
    N_test = 10000
    X_test= np.random.uniform(0,1, N_test*dim).reshape((N_test, dim))
    tensor_shape=[LL for _ in range(dim)]
    tensor_shape[0]=MM
    lr= tensor_prediction( tensor_shape, dim, MM , X_train)

    

    y_lr=np.array([lr.predict(xx) for xx in X_test])
        
        
    y_true = np.array([distribution.density_value(xx) for xx in X_test])
    err_lr = np.linalg.norm(y_lr - y_true,2)**2/np.linalg.norm(y_true,2)**2
        
    lr_rec+=err_lr
        
    y_kde=kernel_density().compute(dim, X_train, X_test)
    #y_kde=1
    print('low rank error = ', lr_rec/(rr+1))
    err_kde = np.linalg.norm(y_kde - y_true,2)**2/np.linalg.norm(y_true,2)**2
    #err_kde=0
    kde_rec+=err_kde
    #print("lr errors =", err_lr, "kde errors =",err_kde)
    print('kde error = ', kde_rec/(rr+1))
 
#full_model=compute_full_tensor(dim, MM, X_train)
#y_full = np.array([full_model.prediction (xx) for xx in X_test])
#err_full=np.linalg.norm(y_full - y_true,2)**2/np.linalg.norm(y_true,2)**2
#print(err_full)
