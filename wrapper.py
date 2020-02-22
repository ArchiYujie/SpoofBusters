#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np


# **read_data_small** is the function to read in the small dataset about 30 MB

# In[10]:


def read_data_small():
    X_train = pd.read_csv("data_small/X_train_small.csv")
    X_test = pd.read_csv("data_small/X_test_small.csv")
    y_train = np.asarray(pd.read_csv("data_small/y_train_small.csv", header=None)[0])
    return X_train, X_test, y_train

X_train, X_test, y_train = read_data_small()

# **read_data_big** is the function to read in the big dataset about 100 MB

# In[11]:


def read_data_big():
    X_train = pd.read_csv("data_big/X_train_big.csv")
    X_test = pd.read_csv("data_big/X_test_big.csv")
    y_train = np.asarray(pd.read_csv("data_big/y_train_big.csv", header=None)[0])
    return X_train, X_test, y_train


# **read_data** is the function to read in the whole dataset about 1.5 G




def read_data():
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = np.asarray(pd.read_csv("data/y_train.csv", header=None)[0])
    return X_train, X_test, y_train


# # Insert Your Code Here

# **detect_spoofying** is the function for training the classifier and classify the results. 
# 
# Here we provide an simple example.

# In[ ]:


### import libraries here ###
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_validate

### code classifier here ###
def format_data(df):
    
    # append numberical columns
    rst = df.loc[:,["price","volume","bestBid","bestAsk",'bestBidVolume',
                    'bestAskVolume','lv2Bid', 'lv2BidVolume','lv2Ask', 
                    'lv2AskVolume', 'lv3Bid', 'lv3BidVolume', 'lv3Ask',
                    'lv3AskVolume']]
    
    # encode the binaries
    rst["isBid"] = df.isBid*1
    rst["isBuyer"] = df.isBuyer*1
    rst["isAggressor"] = df.isAggressor*1
    rst["type"] = (df.type == "ORDER")*1
    rst["source"] = (df.source=="USER")*1
    
    # parse the order id data
    rst["orderId"] = df.orderId.str.split('-').str[-1]
    rst["tradeId"] = df.tradeId.str.split('-').str[-1]
    rst["bidOrderId"] = df.bidOrderId.str.split('-').str[-1]
    rst["askOrderId"] = df.askOrderId.str.split('-').str[-1]
    
    # encode the multiple lable data
    tmp_operation = pd.DataFrame(pd.get_dummies(df.operation), columns=df.operation.unique()[:-1])
    rst = pd.concat([rst, tmp_operation], axis=1)
    tmp_endUserRef = pd.DataFrame(pd.get_dummies(df.endUserRef), columns=df.endUserRef.unique()[:-1])
    rst = pd.concat([rst, tmp_endUserRef], axis=1)
    
    # also feel free to add more columns inferred from data
    # smartly engineered features can be very useful to improve the classification resutls
    rst["timeSinceLastTrade"] = X_train[["timestamp","endUserRef"]].groupby("endUserRef").diff()
    
    return rst

def data_prep(X_train, X_test, y_train):
    # clean up the data
    X_clean = format_data(pd.concat([X_train, X_test]))
    X_clean = X_clean.fillna(-1)
    X_train_clean = X_clean.iloc[:X_train.shape[0],:]
    X_test_clean = X_clean.iloc[X_train.shape[0]:,:]
    X_train_clean_scaled = scale(X_train_clean)
    X_test_clean_scaled = scale(X_test_clean)
    
    return X_train_clean_scaled, X_test_clean_scaled, y_train


# In[ ]:


X_train_clean_scaled, X_test_clean_scaled, y_train = data_prep(X_train, X_test, y_train)
print('HERE')

# In[13]:


def detect_spoofying_logistic(X_train_clean_scaled, X_test_clean_scaled, y_train):
    
    # fit classifier
    clf = LogisticRegression(random_state=0, class_weight='balanced').fit(X_train_clean_scaled, y_train)
    y_train_prob_pred = clf.predict_proba(X_train_clean_scaled)
    y_test_prob_pred = clf.predict_proba(X_test_clean_scaled)
    
    return y_train_prob_pred, y_test_prob_pred


# **score** is the function that we use to compare the results. An example is provided with scoring the predictions for the training dataset. True labels for the testing data set will be supplied to score the predictions for testing dataset.

# Score is based on cohen's kappa measurement. https://en.wikipedia.org/wiki/Cohen%27s_kappa

# In[14]:


from sklearn.metrics import cohen_kappa_score

def score(y_pred, y_true):
    """
    y_pred: a numpy 4d array of probabilities of point assigned to each label
    y_true: a numpy array of true labels
    """
    y_pred_label = np.argmax(y_pred, axis=1)
    return cohen_kappa_score(y_pred_label, y_true)


# ### Optional: k-fold cross validation

# In[32]:


# ### optional: examples of k-fold cross validation ###
# # k-fold cross validation can help you compare the classification models
# from sklearn.model_selection import KFold
# n = 5 # here we choose a 10 fold cross validation
# kf = KFold(n_splits = n)
# X_train, X_test, y_train = read_data_small()
# kf.get_n_splits(X_train)
# print(kf)
# kf_scores = pd.DataFrame(np.zeros([n,2]), columns=["train score", "test score"])
# rowindex = 0
# for train_index, test_index in kf.split(X_train):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     print(X_train.index)
#     print(y_train)
#     X_train_kf, X_test_kf = X_train.iloc[train_index], X_train.iloc[test_index]
#     y_train_kf, y_test_kf = y_train[train_index], y_train[test_index]
#     y_train_prob_pred_kf, y_test_prob_pred_kf = detect_spoofying(X_train_kf, X_test_kf, y_train_kf)
#     score_train_kf = score(y_train_prob_pred_kf, y_train_kf)
#     score_test_kf = score(y_test_prob_pred_kf, y_test_kf)
#     kf_scores.iloc[rowindex, 0] = score_train_kf
#     kf_scores.iloc[rowindex, 1] = score_test_kf
#     rowindex += 1


# In[8]:



# **wrapper** is the main function to read in unzipped data and output a score for evaluation. In addition, the function returns the y probability matrix (both train and test) for grading. More details about submitting format are outlined below.

# In[20]:


def wrapper(spoof_detector):
    # read in data
    X_train, X_test, y_train = read_data_small()
    # or if you have the computational power to work with the big data set, 
    # you can comment out the read_data_samll line and uncomment the following read_data_big
    # X_train, X_test, y_train = read_data_big()
    
    # process the data, train classifier and output probability matrix
    y_train_prob_pred, y_test_prob_pred = spoof_detector(X_train, X_test, y_train)
    
    # score the predictions
    score_train = score(y_train_prob_pred, y_train)
    # score_test = score(y_test_prob_pred, y_test)
    
    # return the scores
    return score_train, y_train_prob_pred, y_test_prob_pred


# Call function wrapper:

# In[27]:


# score_train, y_train_prob_pred, y_test_prob_pred = wrapper(detect_spoofying_logistic)


# Score for training data set is:

# In[28]:


# In[29]:


from sklearn.ensemble import RandomForestClassifier

# def detect_spoofying_rf(X_train, X_test, y_train):
def detect_spoofying_rf(X_train_clean_scaled, X_test_clean_scaled, y_train):
    
#     # clean up the data
#     X_clean = format_data(pd.concat([X_train, X_test]))
#     X_clean = X_clean.fillna(-1)
#     X_train_clean = X_clean.iloc[:X_train.shape[0],:]
#     X_test_clean = X_clean.iloc[X_train.shape[0]:,:]
#     X_train_clean_scaled = scale(X_train_clean)
#     X_test_clean_scaled = scale(X_test_clean)

    # fit classifier
    clf = RandomForestClassifier(max_depth=100, random_state=0, class_weight='balanced').fit(X_train_clean_scaled, y_train)
    y_train_prob_pred = clf.predict_proba(X_train_clean_scaled)
    y_test_prob_pred = clf.predict_proba(X_test_clean_scaled)
    
    return y_train_prob_pred, y_test_prob_pred


# In[30]:


# score_train, y_train_prob_pred, y_test_prob_pred = wrapper(detect_spoofying_rf)


# In[31]:




# In[39]:


from sklearn.ensemble import GradientBoostingClassifier



# def detect_spoofying_gb(X_train, X_test, y_train):
def detect_spoofying_gb(X_train_clean_scaled, X_test_clean_scaled, y_train):
    
#     # clean up the data
#     X_clean = format_data(pd.concat([X_train, X_test]))
#     X_clean = X_clean.fillna(-1)
#     X_train_clean = X_clean.iloc[:X_train.shape[0],:]
#     X_test_clean = X_clean.iloc[X_train.shape[0]:,:]
#     X_train_clean_scaled = scale(X_train_clean)
#     X_test_clean_scaled = scale(X_test_clean)

    original_params = {'n_estimators': 1000, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2,
                   'min_samples_split': 5}
    params = dict(original_params)
    # fit classifier
    clf = GradientBoostingClassifier(**params).fit(X_train_clean_scaled, y_train)
    y_train_prob_pred = clf.predict_proba(X_train_clean_scaled)
    y_test_prob_pred = clf.predict_proba(X_test_clean_scaled)
    
    return y_train_prob_pred, y_test_prob_pred


# In[ ]:



# score_train, y_train_prob_pred, y_test_prob_pred = wrapper(detect_spoofying_gb)


# In[ ]:





# ### LSTM

# In[ ]:


# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
print("START")
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# ### Submission Format
# 
# The classifier function wrote should return a 4d nparray with 4 columns. The columns are corresponding to the class labels: 0, 1, 2, 3. Please see examples below.

# In[12]:


# In[13]:


y_test_prob_pred


# ### Write test results to csv files

# Please rename your file to indicate which data set you are working with. 
# 
# - If you are using the small dataset: *y_train_prob_pred_small.csv* and *y_test_prob_pred_small.csv*
# - If you are using the small dataset: *y_train_prob_pred_big.csv* and *y_test_prob_pred_big.csv*
# - If you are using the original dataset: *y_train_prob_pred.csv* and *y_test_prob_pred.csv*

# In[14]:


pd.DataFrame(y_train_prob_pred).to_csv("y_train_prob_pred.csv")
pd.DataFrame(y_test_prob_pred).to_csv("y_test_prob_pred.csv")

