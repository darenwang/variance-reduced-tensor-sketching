from polynomial import generate_basis_vector_given_coordinates,generate_new_basis_vector
import numpy as np

 



class compute_rank_one:
    #we compute rank 1 tensor given basis_val
        #the rank one tensor is 
        #[p_0(x[0]), p_1(x[0]) , ...  ]\otimes [p_0(x[1]),p_1(x[1]),... ]\otimes [p_0(x[d-1]), p_1(x[d-1]),... ]
    def __init__(self,dim):
        self.dim=dim

    def compute(self,basis_val):
        temp_tensor=1
        
        for dd in range(self.dim-1, -1, -1):
            temp_tensor= np.tensordot( np.reshape(basis_val[dd], (len(basis_val[dd]),1)), [temp_tensor],axes=(1,0))

        #for dd in range(dim):
        #    temp_tensor= np.tensordot([temp_tensor], [basis_val[dd]],axes=[[0],[0]])
        return temp_tensor 


class SVD_adaptive_thresholding:
    def __init__(self):
        pass
    def compute(self, A_temp):
        P,D = np.linalg.svd(A_temp, full_matrices=True, hermitian=False)[:2]
        
        #thresholding the rank 
        #should be the maximum ratio!!
        #we keep at least one singular function

        cur_rank=1
        cur_max_ratio=0
        for rank in range(1,len(D)):
            if  D[rank-1]/D[rank]>cur_max_ratio:
                cur_max_ratio=D[rank-1]/D[rank]
                cur_rank=rank
        #we keep everything  D[1,...,rank-1]
        
        ##################################### this is essential  for debugging 
        #print(D,cur_rank)
        cur_rank=min(cur_rank,3)
        
        #print(D)
        #cur_rank=2
        #print('selected rank =', cur_rank)
        
        #P_transpose gets the colunms into the rows
        P_transpose=P.transpose()
        
        P_basis= P_transpose[: cur_rank]
        #print(P_basis)
        
        
        return P_basis





class tensor:
    def __init__(self, dim,index):
        self.dim=dim
        self.generate_basis_vector_given_coordinates = generate_basis_vector_given_coordinates(dim)
        self.new_coordinate=[ ii  for ii in range(self.dim)]
        self.new_coordinate[0]=index
        self.new_coordinate[index]=0
        self.compute_rank_one=compute_rank_one(dim)


    def compute_tensor(self,x_train,tensor_shape, new_coordinates):

            A_temp=np.zeros(tensor_shape, dtype=float)
            
            for i in range(len(x_train)):
                basis_val=self.generate_basis_vector_given_coordinates.compute_basis_val_new_coordinates(x_train[i],tensor_shape, new_coordinates)

                
                A_temp=A_temp+self.compute_rank_one.compute(basis_val)             
                
            A_temp=A_temp/len(x_train)
            return A_temp





    def compute_range_basis(self,x_train,tensor_shape):

        

        A_temp = self.compute_tensor(x_train,tensor_shape, self.new_coordinate)
        #print(A_temp)
        
 
        A_temp= A_temp.reshape( tensor_shape[0],  -1) 
        return SVD_adaptive_thresholding().compute(A_temp)



############iterations


class iteration:
    def __init__(self, tensor_shape, dim,M,X_train):

        self.compute_rank_one=compute_rank_one(dim)
        self.dim=dim 
        self.M=M
        self.X_train=X_train
        
        
        
        ######compute the basis using the tensor class
        #######self.cur_shape is shape of self.cur_basis
        self.cur_basis=[]
        self.cur_shape=[]
        for dd in range(dim):
            print('coordinate =',dd)
            
            self.cur_basis.append( tensor(dim,dd).compute_range_basis(X_train,tensor_shape) ) 
            self.cur_shape.append(len(self.cur_basis[-1]))

    
    def iteration_compute_P_given_index(self, index):
        #####new coordinate 
        new_coordinates=[ ii  for ii in range(self.dim)]
        new_coordinates[0]=index
        new_coordinates[index]=0
        ########
        shape_at_index=self.cur_shape[index]
        self.cur_shape[index]=self.M
        self.cur_shape[0], self.cur_shape[index]= self.cur_shape[index],self.cur_shape[0]
        
        matrix_at_index=self.cur_basis[index]
        self.cur_basis[index]=np.identity(self.M)
        self.cur_basis[0],self.cur_basis[index] =self.cur_basis[index],self.cur_basis[0]
        
        self.generate_new_basis_vector=generate_new_basis_vector(self.dim, self.M, self.cur_basis)


        A_iterate=np.zeros(self.cur_shape, dtype=float)
        for i in range(len(self.X_train)):
            basis_val=self.generate_new_basis_vector.compute_basis_val_given_coordinates( self.X_train[i], new_coordinates) 

            A_iterate=A_iterate+self.compute_rank_one.compute(basis_val)
            
        A_iterate=A_iterate/len(self.X_train)
        #print(self.A_iterate)

        A_iterate= A_iterate.reshape( self.M,  -1) 



        #########backtrak
        self.cur_shape[0], self.cur_shape[index]= self.cur_shape[index],self.cur_shape[0]
        self.cur_shape[index]=shape_at_index
        self.cur_basis[0],self.cur_basis[index] =self.cur_basis[index],self.cur_basis[0]
        self.cur_basis[index]=matrix_at_index

        return SVD_adaptive_thresholding().compute(A_iterate)




    def compute(self):
        new_basis=[]
        for index in range(self.dim):
            print('iteration coordinate =', index)
            new_basis.append(self.iteration_compute_P_given_index(index))
        
        return new_basis
    
#fun=iteration(tensor_shape, dim,MM,X_train)
#P_new=fun.compute()