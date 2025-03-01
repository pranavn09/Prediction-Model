#!/usr/bin/env python
# coding: utf-8

# 
# # Logistic Regression with Python
# 
# For this lecture we will be working with the [Titanic Data Set from Kaggle](https://www.kaggle.com/c/titanic). This is a very famous data set and very often is a student's first step in machine learning! 
# 
# We'll be trying to predict a classification- survival or deceased.
# Let's begin our understanding of implementing Logistic Regression in Python for classification.
# 
# We'll use a "semi-cleaned" version of the titanic data set, if you use the data set hosted directly on Kaggle, you may need to do some additional cleaning not shown in this lecture notebook.
# 
# ## Import Libraries
# Let's import some libraries to get started!

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## The Data
# 
# Let's start by reading in the titanic_train.csv file into a pandas dataframe.

# In[2]:


train = pd.read_csv('titanic_train.csv')


# In[3]:


train.head()


# # Exploratory Data Analysis
# 
# Let's begin some exploratory data analysis! We'll start by checking out missing data!
# 
# ## Missing Data
# 
# We can use seaborn to create a simple heatmap to see where we are missing data!

# In[4]:


train.isnull()


# In[10]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. We'll probably drop this later, or change it to another feature like "Cabin Known: 1 or 0"
# 
# Let's continue on by visualizing some more of the data! Check out the video for full explanations over these plots, this code is just to serve as reference.

# In[11]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)


# In[78]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[79]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[22]:


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=40)


# In[25]:


train['Age'].hist(bins=30,color='darkred',alpha=0.3)


# In[82]:


sns.countplot(x='SibSp',data=train)


# In[83]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# ____
# ### Cufflinks for plots
# ___
#  Let's take a quick moment to show an example of cufflinks!

# In[84]:


import cufflinks as cf
cf.go_offline()


# In[85]:


train['Fare'].iplot(kind='hist',bins=30,color='green')


# ___
# ## Data Cleaning
# We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation).
# However we can be smarter about this and check the average age by passenger class. For example:
# 

# In[26]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age.

# In[27]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# Now apply that function!

# In[28]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# Now let's check that heat map again!

# In[33]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Great! Let's go ahead and drop the Cabin column and the row in Embarked that is NaN.

# In[30]:


train.drop('Cabin',axis=1,inplace=True)


# In[91]:


train.head()


# In[32]:


train.dropna(inplace=True)


# ## Converting Categorical Features 
# 
# We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.

# In[93]:


train.info()


# In[37]:


pd.get_dummies(train['Embarked'],drop_first=True).head()


# In[38]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[39]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[40]:


train.head()


# In[96]:


train = pd.concat([train,sex,embark],axis=1)


# In[97]:


train.head()


# Great! Our data is ready for our model!
# 
# # Building a Logistic Regression model
# 
# Let's start by splitting our data into a training set and test set (there is another test.csv file that you can play around with in case you want to use all this data for training).
# 
# ## Train Test Split

# In[42]:


train.drop('Survived',axis=1).head()


# In[43]:


train['Survived'].head()


# In[45]:


from sklearn.model_selection import train_test_split


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# ## Training and Predicting

# In[47]:


from sklearn.linear_model import LogisticRegression


# In[48]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[49]:


predictions = logmodel.predict(X_test)


# In[52]:


from sklearn.metrics import confusion_matrix


# In[54]:



accuracy=confusion_matrix(y_test,predictions)


# In[55]:


accuracy


# In[57]:


from sklearn.metrics import accuracy_score


# In[58]:


accuracy=accuracy_score(y_test,predictions)
accuracy


# In[50]:


predictions


# Let's move on to evaluate our model!

# ## Evaluation

# We can check precision,recall,f1-score using classification report!

# In[104]:


from sklearn.metrics import classification_report


# In[105]:


print(classification_report(y_test,predictions))


# Not so bad! You might want to explore other feature engineering and the other titanic_text.csv file, some suggestions for feature engineering:
# 
# * Try grabbing the Title (Dr.,Mr.,Mrs,etc..) from the name as a feature
# * Maybe the Cabin letter could be a feature
# * Is there any info you can get from the ticket?
# 

# In[ ]:




