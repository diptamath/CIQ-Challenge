
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


d1 = pd.read_csv("CIQ-traindata.csv")


# In[4]:


d1.head(3)


# In[5]:


d2 = pd.read_csv("train.csv")


# In[24]:


d3 = pd.read_csv("test.csv")


# In[26]:


d3.head(3)


# In[6]:


d2.head(3)


# In[27]:


dic = {}

qids = list(d2["qid"]) + list(d3["qid"])
qstns = list(d2["question_text"]) + list(d3["question_text"])

for i in range(0,len(qids)) :
    dic[qids[i]] = qstns[i]


# In[28]:


print(list(dic.items())[0])


# In[29]:


lis = []

for i,row in d1.iterrows() :
    #print(row[qid])
    try :
        lis.append(dic[row["qid"]])
    except :
        lis.append('nan')


# In[30]:


print(lis)


# In[31]:


len(d1)


# In[32]:


len(lis)


# In[33]:


d1["Questions"] = lis


# In[34]:


d1.to_csv("ids_and_questions_dataset.csv")


# In[25]:


a = [1,2,3]
b = [4,5,6]
c= a+b
print(c)

