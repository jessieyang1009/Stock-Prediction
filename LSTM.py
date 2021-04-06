#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dropout
from keras.layers import Embedding


# In[40]:


path="D:\\master\\master_1091\\軟性計算\\2474-0415\\2474\\label\\table_label_new.csv"
path2="D:\\master\\master_1091\\軟性計算\\2474-0415\\2474\\label\\table_new.csv"
data = pd.read_csv(path,index_col="news")
table = pd.read_csv(path2,index_col="news")
#將資料切割訓練及測試
X = table
y = np.array(data["label"])
X


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state=101)


# In[42]:


X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train


# In[43]:


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


# In[44]:


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[32]:


regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 125, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 125, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

regressor.summary()


# In[33]:


train_history_LSTM = regressor.fit(x=X_train,  
                          y=y_train, validation_split=0.1,  
                          epochs=100, batch_size=32, verbose=1)


# In[34]:


plt.plot(train_history_LSTM.history['loss'])
plt.plot(train_history_LSTM.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left') 
plt.show()


# In[35]:


plt.plot(train_history_LSTM.history['f1_m'], 'r',label='f1')
plt.plot(train_history_LSTM.history['precision_m'], 'g',label='precision')
plt.plot(train_history_LSTM.history['recall_m'], 'k',label='recall')
plt.xlabel('Epochs')
plt.grid()
plt.legend(loc=1)
plt.show()


# In[36]:


plt.plot(train_history_LSTM.history['acc'], 'k')
plt.ylabel('accuracy')
plt.xlabel('Epochs')
plt.grid()
plt.legend(loc=1)
plt.show()


# In[38]:



df = pd.DataFrame(train_history_LSTM.history)
df


# In[39]:


df.to_csv(r'D:\master\master_1091\軟性計算\期末\LSTM_epochs_100_batch_size=32.csv')


# In[ ]:




