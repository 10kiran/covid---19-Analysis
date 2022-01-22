#!/usr/bin/env python
# coding: utf-8

# Dataset URL

# In[18]:


Url = "https://raw.githubusercontent.com/SR1608/Datasets/main/covid-data.csv"


# # 1. Importing the dataset

# In[61]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ### Assigning Dataset in df

# In[20]:


df = pd.read_csv(Url)


# In[21]:


df


# # 2. High level Data Understanding
# ### a. Find no. of rows and columns in the dataset

# In[25]:


print('Number of Rows :',df.shape[0], '\nNumber of Columns :',df.shape[1])


# ### b. Data types of columns

# In[27]:


print(df.dtypes)


# ### c. Info & Describe of data in dataframe 

# In[28]:


df.info()


# In[29]:


df.describe()


# # 3. Low Level Data Understanding
# ### a. Find count of unique values in location column

# In[31]:


print('Number of Unique locations :',df['location'].nunique())


# ### b. Find which continent has maximum frequency using values counts

# In[32]:


print('Continent with maximun frequency :',df['continent'].value_counts().idxmax())


# ### c. Find maximum & mean value in 'total_cases'

# In[33]:


print("Maximum value in 'total_cases' :",df['total_cases'].max(),
     "\nMean value in 'total_cases':",df['total_cases'].mean())


# ### d.Find 25%, 50% & 75% quartile value in 'total_deaths'

# In[34]:


print("25% Quqrtile in 'total_deaths' :",df['total_deaths'].quantile(0.25),
     "\n50% Quqrtile in 'total_deaths' :",df['total_deaths'].quantile(0.50),
     "\n75% Quqrtile in 'total_deaths' :",df['total_deaths'].quantile(0.75),)


# ### e.Find which continent has maximum 'human_development_index'

# In[36]:


df.groupby('continent')['human_development_index'].max().nlargest(1)


# In[37]:


df['human_development_index'].idxmax()


# ### f. Find which continent has minimum 'gdp_per_capita'.

# In[38]:


df.groupby('continent')['gdp_per_capita'].min().nsmallest(1)


# # 4. Filter the dataframe with only this columns
# ### ['continent','location','date','total_cases','total_deaths','gdp_per_capita','human_development_index'] and update the data frame.

# In[39]:


df=df[['continent','location','date','total_cases','total_deaths','gdp_per_capita','human_development_index']]


# In[40]:


df


# # 5. Data Cleaning
# ### a. Remove all duplicates observations

# In[41]:


df=df.drop_duplicates()


# In[42]:


df


# ### b. Find missing values in all columns
# 

# In[43]:


df.isnull().sum()


# ### c. Remove all observations where continent column value is missing

# In[44]:


df.dropna(subset=['continent'],inplace=True)


# In[45]:


df.isnull().sum()


# ### d. Fill all missing values with 0

# In[46]:


df.fillna(0,inplace=True)


# In[47]:


df.isnull().sum()


# # 6. Date time format :
# ### a. Convert date column in datetime format using pandas.to_datetime

# In[48]:


df.dtypes


# In[49]:


df['date']=pd.to_datetime(df['date'])


# In[50]:


df.dtypes


# ### b. Create new column month after extracting month data from date column.

# In[51]:


df['month']=df['date'].dt.month


# In[52]:


df[df['location']=='India']


# # 7. Data Aggregation:
# ### a. Find max value in all columns using groupby function on 'continent' column
# 

# In[54]:


df.groupby('continent').max().reset_index()


# ### b. Store the result in a new dataframe named 'df_groupby'.
# 

# In[55]:


df_groupby = df.groupby('continent').max().reset_index()


# In[56]:


df_groupby


# # 8. Feature Engineering :
# ### a. Create a new feature 'total_deaths_to_total_cases' by ratio of 'total_deaths' column to 'total_cases'

# In[57]:


df_groupby['total_deaths_to_total_cases'] = df_groupby['total_deaths']/df_groupby['total_cases']


# In[58]:


df_groupby


# # 9. Data Visualization :
# ### a. Perform Univariate analysis on 'gdp_per_capita' column by plotting histogram using seaborn dist plot.

# In[59]:


import warnings
warnings.filterwarnings('ignore')


# In[73]:


displot1 = sns.distplot(df_groupby['gdp_per_capita'],color='green')
displot1.set(title='Univariate analysis on gdp_per_capita')
# Saving/exporting the figure
#displot2.get_figure().savefig('9_a_1.jpg')


# ### b. Plot a scatter plot of 'total_cases' & 'gdp_per_capita'

# In[74]:


scatterplot1=sns.scatterplot(x=df_groupby['total_cases'],y=df_groupby['gdp_per_capita'])
scatterplot1.set(title='Scatter plot of total_cases & gdp_per_capita')

# Saving/exporting the figure
#scatterplot2.get_figure().savefig('9_b_1.jpg')


# ### c. Plot Pairplot on df_groupby dataset.

# In[70]:


pairplot1=sns.pairplot(df_groupby)
pairplot1.fig.suptitle('Pairplot on grouped data',y=1.05,fontsize=20)

#Saving/exporting the figure
pairplot1.savefig('9_c_1.jpg')


# ### d. Plot a bar plot of 'continent' column with 'total_cases'.

# In[75]:


barplot1=sns.catplot(data=df_groupby,x='continent',y='total_cases',kind='bar',aspect=1.5)
barplot1.set(title='Bar plot of continent with total_cases')
# Saving/exporting the figure
#barplot2.savefig('9_d_1.jpg')


# # 10. Save the df_groupby dataframe in your local drive using pandas.to_csv function

# In[76]:


df_groupby.to_csv('COVID-19_grouped_data.csv')


# In[ ]:




