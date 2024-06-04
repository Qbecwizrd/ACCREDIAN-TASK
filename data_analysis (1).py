#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


# In[39]:


pip install imblearn


# In[2]:


df=pd.read_csv('Fraud.csv')


# In[3]:


df


# In[5]:


df.shape


# In[12]:


df.head()


# In[26]:


df.info()


# In[35]:


df_good_transactions=len(df[df['isFraud']==0])
df_bad_transactions=len(df[df['isFraud']==1])


# In[36]:


good_percentage=((df_good_transactions)/(df_bad_transactions+df_good_transactions))*100
bad_percentage=((df_bad_transactions)/(df_bad_transactions+df_good_transactions))*100


# In[37]:


print('percentage of good transactions',good_percentage)
print('percentage of bad transactions',bad_percentage)


# In[48]:


numeric_list=['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud','isFlaggedFraud']


# In[50]:


df_numeric=df[numeric_list]


# In[51]:


df_numeric


# In[53]:


correlation_matrix=df_numeric.corr()


# In[54]:


correlation_matrix


# In[55]:


sns.clustermap(correlation_matrix,cmap='coolwarm')
plt.show()


# SOLVING THE PROBLEM

# encoding the various not numerical atributes of the data

# In[57]:


label_encoder=LabelEncoder()


# In[60]:


obj_list=df.select_dtypes(include='object').columns


# In[61]:


obj_list


# In[63]:


for column in obj_list:
    df[column]=label_encoder.fit_transform(df[column])


# In[64]:


df


# i dont think that nameOrig  and nameDest have much influence on the model

# In[67]:


df=df.drop(['nameOrig','nameDest'],axis=1)


# In the above clusterplot we notice that oldbalanceOrg and newbalanceOrig are highly correlated .Similarly oldbalanceDest and newbalanceDest are also higly correalted and hence i plan to combine them into one clumn to reduce the colinear nature

# In[68]:


df['net_amount_org']=df.apply(lambda x:x['oldbalanceOrg']-x['newbalanceOrig'],axis=1)


# In[69]:


df['net_amount_dest']=df.apply(lambda x:x['oldbalanceDest']-x['newbalanceDest'],axis=1)


# In[73]:


df=df.drop(['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','step'],axis=1)


# In[74]:


df


# In[75]:


correlation_matrix=df.corr()


# In[76]:


sns.clustermap(correlation_matrix,cmap='coolwarm')
plt.show()


# we can notice that the collinearity among the predictor variables have stopped hence we can apply the model now

# In[88]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix


# In[80]:


y=df['isFraud']
X=df.drop('isFraud',axis=1)


# In[81]:


# Assuming 'X' is your feature matrix and 'y' is your target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[82]:


adaboost_clf = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost_clf.fit(X_train, y_train)


# In[83]:


y_pred = adaboost_clf.predict(X_test)


# In[84]:


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))



# In[89]:


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()


# In[91]:


print("TP,FP,TN,FN - Decision Tree")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f'True Positives: {tp}')
print(f'False Positives: {fp}')
print(f'True Negatives: {tn}')
print(f'False Negatives: {fn}')


# In[ ]:




