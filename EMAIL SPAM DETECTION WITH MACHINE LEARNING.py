#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import re
import nltk
 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# In[2]:


import nltk
nltk.download('stopwords')


# In[3]:


df = pd.read_csv("spam.csv", delimiter=",", encoding="latin-1")


# In[4]:


df


# In[5]:


ps=PorterStemmer()
corpus=[]


# In[6]:


for i in range(len(df['v2'])):
    review=re.sub('[^a-zA-Z]',' ',df['v2'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)


# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500)
X=cv.fit_transform(corpus).toarray()


# In[8]:


y=pd.get_dummies(df['v1'])
y=y.iloc[:,:1]


# In[9]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[10]:


from sklearn.ensemble import RandomForestClassifier


# In[11]:


rf=RandomForestClassifier()
rf.fit(X_train,y_train)
rf.score(X_train,y_train)
y_pred=rf.predict(X_test)


# In[12]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(accuracy_score(y_pred,y_test))


# In[ ]:




