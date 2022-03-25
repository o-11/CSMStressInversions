## an example of calculating a composite posterior:
import h5py
import numpy as np

#Modify------------------------
pts1=np.load('')
pts2=np.load('')
cts1=np.load('')
cts2=np.load('')

name1=''
name2=''
#End----------------------------

#Functions------
def posterior_operations(dict1,dict2,operation):
        """
        takes two posteriors organized as dictionaries (key: lattice point location or id, value: number of
 times visited)
        performs operation in order of dictionaries inputted (e.g. "dict1-dict2")
        """
        if operation=='sum':
                return {key: dict1.get(key,0) + dict2.get(key,0) for key in {**dict1,**dict2}.keys()}

        if operation=='subtract':
                return {key: dict1.get(key,0) - dict2.get(key,0) for key in {**dict1,**dict2}.keys()}

        if operation=='multiply':
                return {key: dict1.get(key,0) * dict2.get(key,0) for key in {**dict1,**dict2}.keys()}

#--------------

len_pts1=len(pts1)
len_pts2=len(pts2)
freq1={str(pts1[i]): cts1[i]*(1/(len_pts1)) for i in range(len_pts1)}
freq2={str(pts2[i]): cts2[i]*(1/(len_pts2)) for i in range(len_pts2)}

del pts1; del pts2; del cts1; del cts2

#to check dictionary size, use:
##import sys
##sys.getsizeof(dicta) #in bytes

freq_sum=posterior_operations(freq1,freq2,"sum") #composite posterior (mutually exclusive)
freq_j=posterior_operations(freq1,freq2,"multiply") #joint posterior
freq_nme=posterior_operations(freq_sum,freq_j,"subtract") #composite posterior (not mutually exclusive)

#save all three for plotting and further operations:
list_sum=list(freq_sum.items())
arr_sum=np.array(list_sum)
del list_sum
list_j=list(freq_j.items())
arr_j=np.array(list_j)
del list_j
list_nme=list(freq_nme.items())
arr_nme=np.array(list_nme)
arr_sum_=np.zeros((len(arr_sum),6));arr_j_=np.zeros((len(arr_j),6));arr_nme_=np.zeros((len(arr_nme),6));
for i in range(len(arr_sum)):
    arr_sum_[i]=np.hstack((list(map(float,arr_sum[i][0][1:-1].split( ))),float(arr_sum[i][1])))
del arr_sum

for i in range(len(arr_j)):
    arr_j_[i]=np.hstack((list(map(float,arr_j[i][0][1:-1].split( ))),float(arr_j[i][1])))
del arr_j

for i in range(len(arr_nme)):
    arr_nme_[i]=np.hstack((list(map(float,arr_nme[i][0][1:-1].split( ))),float(arr_nme[i][1])))
del arr_nme

freqs=h5py.File(name + 'compositeposteriorfiles.h5','a')
ds_sum=freqs.create_dataset('fc_me',data=arr_sum_)
ds_j=freqs.create_dataset('fj',data=arr_j_)
ds_nme=freqs.create_dataset('fc_nme',data=arr_nme_)
freqs.close()
