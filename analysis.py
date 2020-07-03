#!/usr/bin/env python
# coding: utf-8

# ## Import SEG_SGEMM Data as int64

# In[275]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_as_int64 = np.genfromtxt("./로그 데이터/SEG_SGEMM_result.txt", delimiter="\n", dtype=np.int64).reshape(-1, 1)
data_as_int64 = pd.DataFrame(data=data_as_int64, columns=["SEG"])
data_as_int64


# ## Import SEG_SGEMM Data as String (object)  
# Since this is not regression problem, regard each of data as string, not numerics

# In[276]:


original = pd.read_csv("./로그 데이터/SEG_SGEMM_result.txt")
original.columns = ["SEG"]
split_index = 20000
data = original.copy()
train_set = data[:split_index]
test_set = data[split_index:]
test_set.drop(test_set.tail(1).index, inplace=True)
train_set, test_set


# ## Create Label and Split into Train/Test using split_index 

# In[277]:


label = original.copy()
label.drop(original.head(1).index, inplace=True)
label = label.reset_index(drop=True)
train_label = label[:split_index]
test_label = label[split_index:]
train_label, test_label


# ## Histogram of SEG using value_counts()
# Since there are too many categories, unable to draw Histogram.

# In[278]:


data_counts = train_label["SEG"].value_counts().to_frame("SEG_counts")
data_counts


# ## Number of counts over threshold  
# True if number of count is greater than threshold_num_of_label, False otherwise.  
# Interpretation : There are only 20 Categories that contain more than 100 data.  
# Use threshold_num_of_label as Hyper-Parameter

# In[279]:


threshold_num_of_label = 100
data_counts_over_threshold = (data_counts > threshold_num_of_label)["SEG_counts"]
data_counts_over_threshold.value_counts()

labels_over_threshold = data_counts_over_threshold.index[data_counts_over_threshold == True]
labels_over_threshold


# ## Row indexes of data to be considered as "others"

# In[282]:


train_label_mapped = train_label["SEG"].isin(labels_over_threshold)
train_label_mapped_index = train_label_mapped.index[train_label_mapped == False]
train_label_mapped_index


# ## Represent data as "-1" (others)

# In[283]:


train_label.loc[train_label_mapped_index] = "-1"
train_label["SEG"].value_counts()


# In[284]:


train_label


# In[285]:


d = train_label["SEG"].value_counts().to_frame("SEG_counts")
d.plot(kind='bar')


# In[299]:


train_dataset = pd.concat([train_set, train_label], axis=1)
train_dataset.columns = ["feature", "label"]
train_set, train_label


# In[314]:


import tensorflow as tf
from tensorflow import keras

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

    if single_step:
        labels.append(target[i+target_size])
    else:
        labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

TRAIN_SPLIT = 18000
past_history = 10
future_target = 10
STEP = 1

#x_train, y_train = multivariate_data(train_set.values, train_label.values, 0, TRAIN_SPLIT, past_history, future_target, STEP, single_step=True)
#x_val, y_val = multivariate_data(train_set.values, train_label.values, TRAIN_SPLIT, None, past_history, future_target, STEP, single_step=True)


# In[364]:


x_train = train_set[:TRAIN_SPLIT].values.reshape(-1, 1, 1)
y_train = train_label[:TRAIN_SPLIT].values.reshape(-1, 1, 1)

x_val = train_set[TRAIN_SPLIT:].values.reshape(-1, 1, 1)
y_val = train_label[TRAIN_SPLIT:].values.reshape(-1, 1, 1)
x_train, y_train


# In[373]:


BUFFER_SIZE = 100000
BATCH_SIZE = 256

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.cache().batch(BATCH_SIZE).repeat()


# In[379]:


for x, y in train_data.take(1):
    print(x.shape)
    print(y.shape)


# In[384]:


single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(32, input_shape=(256, 1)))
single_step_model.add(tf.keras.layers.Dense(1))

single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')


# In[386]:


EVALUATION_INTERVAL = 200
EPOCHS = 10

single_step_history = single_step_model.fit(train_data, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data,
                                            validation_steps=50)


# In[ ]:




