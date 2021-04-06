#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import matplotlib.pyplot as plt


# In[15]:


path="/Users/yangyichen/Desktop/新聞情感分析/table_label.csv"
path2="/Users/yangyichen/Desktop/新聞情感分析/table.csv"
data = pd.read_csv(path,index_col="news")
table = pd.read_csv(path2,index_col="news")
#將資料切割訓練及測試
X = table
y = np.array(data["label"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state=101)


# In[16]:


from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# In[17]:


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[18]:


print(len(y_train))


# In[19]:


# 在 Keras 裡面我們可以很簡單的使用 Sequential 的方法建建立一個 Model
model = Sequential()
# 加入 hidden layer-1 of 512 neurons 並指定 input_dim 為 784
model.add(Dense(125, input_dim=20))
# 使用 'relu' 當作 activation function
model.add(Activation('relu'))
# 加入 hidden layer-2 of 256 neurons
model.add(Dense(256))
# 使用 'relu' 當作 activation function
model.add(Activation('relu'))
# 加入 hidden layer-3 of 128 neurons
model.add(Dense(256))
# 使用 'relu' 當作 activation function
model.add(Activation('relu'))
# 加入 hidden layer-4 of 128 neurons
model.add(Dense(125))
# 使用 'relu' 當作 activation function
model.add(Activation('relu'))
# 加入 output layer of 10 neurons
model.add(Dense(units=1))
# 使用 'softmax' 當作 activation function
model.add(Activation('sigmoid'))

# 定義訓練方式  
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc',f1_m,precision_m, recall_m])
model.summary()


# In[20]:


# 開始訓練  
train_history = model.fit(x=X_train,  
                          y=y_train, validation_split=0.1,  
                          epochs=200, batch_size=32, verbose=1)


# In[21]:


print(train_history.history.keys())


# In[23]:


# summarize history for accuracy
plt.plot(train_history.history['acc'])
plt.plot(train_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss 
plt.plot(train_history.history['loss']) 
plt.plot(train_history.history['val_loss']) 
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left') 
plt.show()
# summarize history for others
plt.plot(train_history.history['f1_m']) 
plt.plot(train_history.history['precision_m']) 
plt.plot(train_history.history['recall_m']) 
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['f1', 'precision','recall'], loc='upper left') 
plt.show()


# In[ ]:




