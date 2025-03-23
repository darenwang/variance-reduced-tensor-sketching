from polynomial import polynomial
import numpy as np

 
class generate_basis_vector:
    def __init__(self):
        self.basis=polynomial()

    #basis_val[d][m] is the m-th basis evaluated at x[d]
    def compute_basis_val(self, dim,x,tensor_shape):
        
        basis_val=[self.basis.compute_single_x_all_degree(tensor_shape[d],x[d]) for d in range(dim)]

        return basis_val
    def compute_basis_val_new_coordinates(self,dim,x,tensor_shape, new_coordinates):
        
        basis_val=[self.basis.compute_single_x_all_degree(tensor_shape[d],x[new_coordinates[d]]) for d in range(dim)]

        return basis_val



######################



class tensor:
    def __init__(self, dim,index):
        self.dim=dim
        self.generate_x_tensor= generate_basis_vector()
        self.new_coordinate=[ ii  for ii in range(self.dim)]
        self.new_coordinate[0]=index
        self.new_coordinate[index]=0


    def compute_rank_one(self,basis_val):
        #we compute rank 1 tensor given basis_val
        #the rank one tensor is 
        #[p_0(x[0]), p_1(x[0]) , ...  ]\otimes [p_0(x[1]),p_1(x[1]),... ]\otimes [p_0(x[d-1]), p_1(x[d-1]),... ]
        temp_tensor=1
        
        for dd in range(self.dim-1, -1, -1):
            temp_tensor= np.tensordot( np.reshape(basis_val[dd], (len(basis_val[dd]),1)), [temp_tensor],axes=(1,0))

        #for dd in range(dim):
        #    temp_tensor= np.tensordot([temp_tensor], [basis_val[dd]],axes=[[0],[0]])
        return temp_tensor 
####### 

    def compute_tensor(self,x_train,tensor_shape, new_coordinates):

            A_temp=np.zeros(tensor_shape, dtype=float)
            
            for i in range(len(x_train)):
                basis_val=self.generate_x_tensor.compute_basis_val_new_coordinates(self.dim,x_train[i],tensor_shape, new_coordinates)

                
                A_temp=A_temp+self.compute_rank_one(basis_val)             
                
            A_temp=A_temp/len(x_train)
            return A_temp





    def compute_range_basis(self,x_train,tensor_shape):
        #index is the x, everything else is y

        

        A_temp = self.compute_tensor(x_train,tensor_shape, self.new_coordinate)
        #print(A_temp)
        
        #dim_right= int(np.prod(tensor_shape[1:]) )
        #A_temp= A_temp.reshape( tensor_shape[0], dim_right ) 
        A_temp= A_temp.reshape( tensor_shape[0],  -1) 
        
        #print(A_temp)
        

        P,D = np.linalg.svd(A_temp, full_matrices=False, hermitian=False)[:2]
        #print(D)
        
        #thresholding the rank 
        #should be the maximum ratio!!
        #we keep at least one singular function
        cur_rank=1
        cur_max_ratio=0
        #for rank in range(1,len(D)):
        for rank in range(1,len(D)):
            if  D[rank-1]/D[rank]>cur_max_ratio:
                cur_max_ratio=D[rank-1]/D[rank]
                cur_rank=rank
        #we keep everything  D[1,...,rank-1]
        #print chosen_rank to ensure that the threshold we choose is good
        
        ##################################### this is essential  for debugging 
        ###omitted for now to run large experiments
        #print(D,cur_rank)
        cur_rank=min(cur_rank,3)
        if len(D)==2:
            cur_rank = 2
        #print(D)
        #print('selected rank =', cur_rank)
        
        #P_transpose gets the colunms into the rows
        P_transpose=P.transpose()
        

        P_basis= P_transpose[: cur_rank]
        #print(P_basis)
        
        
        return P_basis

 
