
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division

import os
import time
import numpy as np
import pandas as pd
import gensim
from tqdm import tqdm
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.stem.lancaster import LancasterStemmer
lc = LancasterStemmer()
from nltk.stem import SnowballStemmer
sb = SnowballStemmer("english")
import gc
from nltk.stem import WordNetLemmatizer   
lemmatizer = WordNetLemmatizer() 


# In[2]:


spell_model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M/wiki-news-300d-1M.vec')
words = spell_model.index2word
w_rank = {}

for i,word in enumerate(words):
    w_rank[word] = i
    
WORDS = w_rank


# In[3]:


# Use fast text as vocabulary
def words(text): 
    return re.findall(r'\w+', text.lower())

def P(word): 
    "Probability of `word`."
    # use inverse of rank as proxy
    # returns 0 if the word isn't in the dictionary
    return - WORDS.get(word, 0)

def correction(word): 
    "Most probable spelling correction for word."
    "correction('quikly') returns quickly    correction('israil') returns israel"
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def singlify(word):
    return "".join([letter for i,letter in enumerate(word) if i == 0 or letter != word[i-1]])


# In[4]:


obscene_words = ['sex','fuck','shit','cunt','gay','lesbian','ass','pussy','dick','penis','vagina','asshole','fap','porn',                 'masturbate','sperm','semen','pregnate','impregnate','boobs','getting laid','get laid','bitch','undress','castrate',                 'castration','incest','sexual','rape','hooker','slut','prostitute','panty','bikini','underwear',                'dildo','breast','transgender','homosexual','anal','butt','bra','paedophilo','']


# In[9]:


def chk_words(s) :
    flag = 0
    
    s=s.split()
    for w in s :
        #print(w + "##")
        if(flag == 1) :
            #print(flag)
            break
            
        if(w in obscene_words) :
            flag = 1
            continue
            
        word = w.lower()
        if(word in obscene_words) :
            flag = 1
            continue
            
        word = w.upper() 
        if(word in obscene_words) :
            flag = 1
            continue
            
        word = w.capitalize() 
        if(word in obscene_words) :
            flag = 1
            continue
            
        word = ps.stem(w)
        if(word in obscene_words) :
            flag = 1
            continue
            
        word = lc.stem(w)
        if(word in obscene_words) :
            flag = 1
            continue
            
        word = sb.stem(w)
        if(word in obscene_words) :
            flag = 1
            continue
            
        if(len(w) > 1) :
            word = correction(w)
            if(word in obscene_words) :
                flag = 1
                continue
    
        word = lemmatizer.lemmatize(w)
        if(word in obscene_words) :
            flag = 1
            continue
                

    return flag


# In[13]:


sent = "Can Aman pregnate a cow?"

print(chk_words(sent))

