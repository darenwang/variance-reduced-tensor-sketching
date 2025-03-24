import numpy as np

class indexing:
    def __init__(self) -> None:
        pass
    def generate(self,max_level):
        seen={}
        for index in range(2**max_level):
            temp=len(bin(index+1))-2
            seen[index+1]=(temp-1,index+1-2**(temp-1))
        return seen

#indexing().generate(5)


class wavelet:
    def __init__(self) -> None:
        self.seen=indexing().generate(6)
        self.power=[np.power(2,nn/2) for nn in range(30)]
        #print(self.power)
    def generator(self,x):
        if x>1 or x<0:
            return 0
        if x<0.5:
            return 1
        return -1
    def compute(self,l_index,x):
        if l_index==0:
            return 1
        level, k= self.seen[l_index]
        #print(level, k)
        #print(self.power)
        #print(self.seen)
        return  self.power[level]*self.generator((self.power[level]**2)*x-k)

    def compute_single_x_all_basis(self,  number_of_basis,x_input):
        
        return np.array([ self.compute(ii, x_input) for ii  in range(number_of_basis)  ])

basis=wavelet()
basis.compute_single_x_all_basis(10, 0.3)


class wavelet_new_basis:
    def __init__(self,M) -> None:
        self.M=M
        self.wavelet=wavelet()
        
        
        
        
    def compute_single_x_all_new_basis(self ,U,x_input):
        vec_temp= self.wavelet.compute_single_x_all_basis(self.M, x_input)
        #print(vec_temp)
        return np.inner( vec_temp, U)

#wavelet_new_basis(3).compute_single_x_all_new_basis([1,0,0], 0.4)
