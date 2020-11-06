# In[50]:


import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


# In[51]:


#get training data
data=np.genfromtxt('train.csv', dtype=str,delimiter=',')
data_label=np.array([],dtype=int)
data_pic=np.array([[]],dtype=float)
temp_lab=[]
temp_pic=[]
for i in range(28709):
    if i ==0:
        pass
    else:
        temp_lab.append(int(data[i][0]))
        temp=[]
        for j in data[i][1].split(" "):
            temp.append(int(j))
        temp_pic.append(temp)
data_label=np.array(temp_lab,dtype=int)
data_pic=np.array(temp_pic,dtype=float,ndmin=2)
#get testing data
datat=np.genfromtxt('test.csv', dtype=str,delimiter=',')
datat_label=np.array([],dtype=int)
datat_pic=np.array([[]],dtype=float)
tempt_lab=[]
tempt_pic=[]
for i in range(7178):
    if i ==0:
        pass
    else:
        tempt_lab.append(int(datat[i][0]))
        tempt=[]
        for j in datat[i][1].split(" "):
            tempt.append(int(j))
        tempt_pic.append(tempt)
datat_label=np.array(tempt_lab,dtype=int)
datat_pic=np.array(tempt_pic,dtype=float,ndmin=2)
datat_pic_normalize=datat_pic/255


# In[53]:


import keras as kr
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import SGD,adam
from keras.utils import np_utils
data_pic_normalize=data_pic/255
data_label_cate=np_utils.to_categorical(data_label,7)
module=Sequential()
module.add(Dense(input_dim=48*48,units=500,activation='relu'))
module.add(Dense(units=500,activation='relu'))
module.add(Dense(units=500,activation='relu'))
module.add(Dense(units=500,activation='relu'))
module.add(Dense(units=7,activation='softmax'))
module.compile(loss='categorical_crossentropy',optimizer='adam')


# In[55]:


module.fit(data_pic_normalize,data_label_cate,batch_size=1000,epochs=10)


# In[56]:


result=module.predict(datat_pic_normalize)


# In[58]:


def classfy(list):
    result=0
    temp=0.00
    for i in range(len(list)):
        if temp<list[i]:
            temp=list[i]
            result=i
    return result
def out(classfy):
    if classfy==0:
        print("生氣")
    if classfy==1:
        print("厭惡")
    if classfy==2:
        print("恐懼")
    if classfy==3:
        print("高興")
    if classfy==4:
        print("難過")
    if classfy==5:
        print("驚訝")
    if classfy==6:
        print("中立")
def show(int):
    plt.imshow(datat_pic[int].reshape(48,48), cmap='gray')
    out(classfy(result[int]))

#predict one of test date
show(504)


# In[12]:


#PCA practice

import sklearn
from sklearn.decomposition import PCA,NMF
pca = PCA(n_components=20)
r_pca=pca.fit_transform(data_pic)


# In[27]:


colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
plt.imshow(pca.components_[9].reshape(48,48), cmap='gray')
for i in range(len(colors)):
    x = r_pca[:, 0][data_label == i]
    y = r_pca[:, 1][data_label == i]
    plt.scatter(x, y, c=colors[i],s=1)
#plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("PCA Scatter Plot")
plt.show()


# In[16]:


#NMF
nmf = NMF(n_components=20)
r_nmf=nmf.fit_transform(data_pic)


# In[21]:


plt.imshow(nmf.components_[4].reshape(48,48), cmap='gray')


# In[ ]:




