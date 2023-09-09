#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Installation
get_ipython().system('pip install git+https://github.com/rcmalli/keras-vggface.git')
get_ipython().system('pip install upgrade tensorflow')
get_ipython().system('pip install keras_applications')
get_ipython().system('pip install keras_preprocessing')
get_ipython().system('pip show tensorflow')


# In[2]:


#Importing the libraries
from keras.preprocessing.image import img_to_array, load_img, array_to_img
from keras_vggface.utils import preprocess_input
from tensorflow import keras
from keras_vggface.vggface import VGGFace
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import gc
import glob
from collections import defaultdict
from itertools import combinations
import cv2
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from random import choice, sample
from keras.metrics import AUC, Accuracy
from keras.layers import Input, Dense, Flatten, Subtract, Dropout, Multiply
from keras.layers import Lambda, Concatenate, GlobalMaxPool2D, GlobalAvgPool2D
from keras.models import Model 
from keras.regularizers import l2
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras.losses import binary_crossentropy
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import threading


# In[3]:


#Pre processing image - function defination
IMG_DIM = (224,224,3)
def read_img(path):
  return preprocess_input( img_to_array( load_img(path, target_size=IMG_DIM) ),version=2)


# In[4]:


# Extracting images 
allPhotos1 = defaultdict(list)
for family in glob.glob("/content/drive/MyDrive/CS 271P Project/train-faces/*"):
  for mem in glob.glob(family+'/*'):
    for photo in glob.glob(mem+'/*'):
      allPhotos1[mem].append(photo)
ppl = list(allPhotos1.keys())
len(ppl)


# In[5]:


# Populating the input dataset
data = pd.read_csv('/content/drive/MyDrive/CS 271P Project/train-pairs.csv')
data.p1 = data.p1.apply( lambda x: '/content/drive/MyDrive/CS 271P Project/train-faces/'+x )
data.p2 = data.p2.apply( lambda x: '/content/drive/MyDrive/CS 271P Project/train-faces/'+x )
data = data[data.columns[:2]]
data['related'] = 1 
print(data.shape)
data.head()


# In[6]:


# Reading all image files
families = glob.glob('/content/drive/MyDrive/CS 271P Project/train-faces/*')
p1 = []; p2 = []
for f1,f2 in combinations(families,2) :
  for _p1 in glob.glob( '{}/*'.format(f1) ):
    for _p2 in glob.glob( '{}/*'.format(f2) ):
      p1.append( _p1 ); p2.append( _p2 );


# In[7]:


# labelling unreleated image pairs
temp = pd.DataFrame({'p1':p1,'p2':p2,'related':np.zeros( (len(p1),) ,dtype=np.int32)})


# In[8]:


temp.shape


# In[9]:


# Combining unrelated and related dataset
temp = temp.sample(n=data.shape[0]*2)
data = data.append(temp).sample(frac=1.).reset_index().drop(['index'],axis=1)
print(data.shape)
data.head()


# In[10]:


#validating image path in the dataset
data = data[ ( (data.p1.isin(ppl)) & (data.p2.isin(ppl)) ) ]
data = [ ( x[0], x[1]  ) for x in data.values ]
len(data)


# In[11]:


# Checking first two pairs in the data file
f, ax = plt.subplots(2, 2, figsize=(4, 4))
batch = sample(data,2)
for i,j in [(0,0),(0,1),(1,0),(1,1)]:
 ax[i][j].imshow( cv2.imread( choice(allPhotos1[batch[i][j]]) ) )


# In[12]:


# Splitting training and testing dataset
train = [ x for x in data if 'F09' not in x[0]  ]
val = [ x for x in data if 'F09' in x[0]  ]
len(train), len(val)


# In[13]:


del data; gc.collect();


# In[14]:


# Defining image function
def getImages(p1,p2):
    p1 = read_img(choice(allPhotos1[p1]))
    p2 = read_img(choice(allPhotos1[p2]))
    return p1,p2
# Defining function for getting images in batches
def getMiniBatch(batch_size=16, data=train):
  p1 = []; p2 = []; Y = []
  batch = sample(data, batch_size//2)
  for x in batch:
    _p1, _p2 = getImages(*x)
    p1.append(_p1);p2.append(_p2);Y.append(1)
  while len(Y) < batch_size:
    _p1,_p2 = tuple(np.random.choice(ppl,size=2, replace=False))
    if (_p1,_p2) not in train+val and (_p2,_p1) not in train+val:
      _p1,_p2 = getImages(_p1,_p2)
      p1.append(_p1);p2.append(_p2);Y.append(0) 
  return [np.array(p1),np.array(p2)], np.array(Y)


# In[15]:


# Defining VGG model using resnet50
# vggface = VGGFace(model='senet50', include_top=False)
vggface = VGGFace(model='resnet50', include_top=False)
for layer in vggface.layers[:-60]:
  layer.trainable=False


# In[16]:


# defining function
def initialize_bias(shape, name=None, dtype=None):
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)
  
def initialize_weights(shape, name=None, dtype = None):
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)


# In[17]:


IMG_DIM = (224,224,3)
def auc(y_true, y_pred):
    return tf.compat.v1.py_function(roc_auc_score, (y_true, y_pred), tf.double)

#creating input pairs for Siamese network
left_input = Input(IMG_DIM)
right_input = Input(IMG_DIM)

x1 = vggface(left_input)
x2 = vggface(right_input)

x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

fc = Dense(100,activation='relu',kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias)
x1 = fc(x1)
x2 = fc(x2)

# |h1-h2|
x3 = Lambda(lambda tensors : K.abs(tensors[0] - tensors[1]))([x1, x2])

# |h1-h2|^2
x4 = Lambda(lambda tensor  : K.square(tensor))(x3)

# h1*h2
x5 = Multiply()([x1, x2])
# h1^2
x6 = Lambda(lambda tensor  : K.square(tensor))(x1)
# h2^2
x7 = Lambda(lambda tensor  : K.square(tensor))(x2)

# |h1-h2|^2 + h1*h2
x = Concatenate(axis=-1)([x4,x5])

x = Dense(100,activation='relu',kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias)(x)
x = Dropout(0.1)(x)

prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(x)

siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

optimizer = Adam(1e-5)

siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy',auc])


# In[18]:


#Training the model
reducelr = ReduceLROnPlateau(monitor='val_loss', mode='min',patience=6,factor=0.1,verbose=1)

model_checkpoint  = ModelCheckpoint('model_best_checkpoint.h5', save_best_only=True,
                                    save_weights_only=True, monitor='val_auc', mode='max', verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')

callbacks_list = [reducelr, model_checkpoint, early_stopping]

def Generator(batch_size, data ):
  while True:
    yield getMiniBatch(batch_size=batch_size, data=data)

#creating input variables for the model
train_gen = Generator(batch_size=16,data=train)
val_gen = Generator(batch_size=16,data=val)

#fitting the model
history = siamese_net.fit_generator( train_gen, steps_per_epoch=75, epochs=75, 
                          validation_data=val_gen, validation_steps=5, use_multiprocessing=True,
                          verbose=1,callbacks=callbacks_list, workers=4)


# In[19]:


# Plotting the graphs for accuracy, loss, AUC
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
t = f.suptitle('Siamese Net Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)
epoch_list = history.epoch

ax1.plot(epoch_list, history.history['accuracy'], label='Train_ Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, epoch_list[-1], 5))
ax1.set_ylabel('Accuracy Value');ax1.set_xlabel('Epoch');ax1.set_title('Accuracy')
ax1.legend(loc="best");ax1.grid(color='gray', linestyle='-', linewidth=0.5)

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, epoch_list[-1], 5))
ax2.set_ylabel('Loss Value');ax2.set_xlabel('Epoch');ax2.set_title('Loss')
ax2.legend(loc="best");ax2.grid(color='gray', linestyle='-', linewidth=0.5)

ax3.plot(epoch_list, history.history['auc'], label='Train AUC')
ax3.plot(epoch_list, history.history['val_auc'], label='Validation AUC')
ax3.set_xticks(np.arange(0, epoch_list[-1], 5))
ax3.set_ylabel('AUC');ax3.set_xlabel('Epoch');ax3.set_title('AUC')
ax3.legend(loc="best");ax3.grid(color='gray', linestyle='-', linewidth=0.5)


# In[20]:


siamese_net.load_weights('model_best_checkpoint.h5')


# In[ ]:




