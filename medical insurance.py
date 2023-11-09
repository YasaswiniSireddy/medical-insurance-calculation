#!/usr/bin/env python
# coding: utf-8

# In[110]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import warnings; warnings.simplefilter('ignore')


# In[111]:


insurance_dataset = pd.read_csv('insurance.csv')


# In[112]:


insurance_dataset.head()


# In[113]:


insurance_dataset.shape


# In[114]:


insurance_dataset.info()


# In[115]:


insurance_dataset.isnull().sum()


# In[116]:


insurance_dataset.describe()


# # cleaning the data by converting string values into Numerical values so that they can be used easily to compare

# In[117]:


clean_data = {'sex': {'male' : 0 , 'female' : 1} ,
                 'smoker': {'yes': 1 , 'no' : 0},
                   'region' : {'northwest':0, 'northeast':1,'southeast':2,'southwest':3}
               }
data_copy = insurance_dataset.copy()
data_copy.replace(clean_data, inplace=True)


# In[118]:


data_copy.head(5)


# In[119]:


data_copy.describe()


# In[120]:


corr = data_copy.corr()
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr,cmap='BuPu',annot=True,fmt=".2f")
plt.title("Dependencies of Medical Charges")
plt.show()


# # From the above co-relation we can conclude that somker,age,bmi are more related to charges

# In[123]:


insurance_dataset['sex'].value_counts()


# In[122]:


# Gender column
plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=insurance_dataset)
plt.title('Sex Distribution')
plt.show()


# In[126]:


insurance_dataset['children'].value_counts()


# In[165]:


# children column
plt.figure(figsize=(6,6))
sns.countplot(x='children', data=insurance_dataset,hue='sex',palette='husl')
plt.title('Children')
plt.show()


# In[129]:


insurance_dataset['smoker'].value_counts()


# In[161]:


# smoker column
plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=insurance_dataset)
plt.title('smoker')
plt.show()


# In[133]:


insurance_dataset['region'].value_counts()


# In[159]:


# region column
plt.figure(figsize=(6,6))
sns.countplot(x='region', data=data_copy,hue='sex',palette='Blues')
plt.title('region')
plt.show()


# In[128]:


plt.figure(figsize=(10,7))
plt.title('Smoker vs Charge')
sns.barplot(x='smoker',y='charges',data=data_copy,palette='Blues',hue='sex')


# In[131]:


plt.figure(figsize=(10,7))
plt.title('BMI vs Charge')
sns.scatterplot(x='bmi',y='charges',data=data_copy,palette='Reds',hue='sex')


# In[168]:


plt.figure(figsize=(10,7))
plt.title('age vs Charge')
sns.scatterplot(x='age',y='charges',data=data_copy,palette='Set3',hue='sex')


# In[148]:


sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['age'])
plt.title('Age Distribution')
plt.show()


# In[149]:


# bmi distribution
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['bmi'])
plt.title('BMI Distribution')
plt.show()


# In[150]:


# distribution of charges value
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['charges'])
plt.title('Charges Distribution')
plt.show()


# In[135]:


# encoding sex column
insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)

3 # encoding 'smoker' column
insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

# encoding 'region' column
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)


# In[136]:


X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']


# In[137]:


print(X)


# In[138]:


print(Y)


# In[139]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[140]:


print(X.shape, X_train.shape, X_test.shape)


# In[141]:


# loading the Linear Regression model
regressor = LinearRegression()


# In[142]:


regressor.fit(X_train, Y_train)


# In[143]:


# prediction on training data
training_data_prediction =regressor.predict(X_train)


# In[144]:


# R squared value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared vale : ', r2_train)


# In[145]:


# prediction on test data
test_data_prediction =regressor.predict(X_test)


# In[146]:


# R squared value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared vale : ', r2_test)


# In[ ]:





# In[157]:


input_data = (31,1,25.74,0,1,0)

# changing input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
input_data_reshaped


# In[155]:


prediction = regressor.predict(input_data_reshaped)
print(prediction)

print('The insurance cost is', prediction[0])


# In[ ]:





# In[ ]:




