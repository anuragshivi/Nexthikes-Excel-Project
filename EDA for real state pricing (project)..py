#!/usr/bin/env python
# coding: utf-8

# ##### "This projectexplores the dynamics of house pricing using EDA techniques to extract bussiness insights "

# ### 1 . Import Required Libraries

# In[68]:


pip show seaborn


# In[69]:


pip install --upgrade fsspec


# In[70]:


conda update fssfec -y


# In[71]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("seaborn:", sns.__version__)


# ### 2 . Load the Dataset

# * Load dataset

# In[72]:


df = pd.read_csv(r"C:\Users\Pc\Downloads\housing_data.csv")


# * show first 5 rows

# In[73]:


print(df.head())


# * check shape & info

# In[74]:


print("Shape:", df.shape)
print(df.info())


# ### 3 . Data Cleaning 

# In[79]:


# Check missing values 
print(df.isnull().sum())


# In[80]:


# Fill the missing numerical values with median 

df.fillna(df.median(numeric_only=True),
inplace=True)


# In[81]:


# Fill missing categorical values with mode 
df.fillna(df.mode().iloc[0], inplace=True)


# In[82]:


# Remove dublicates if any 
df.drop_duplicates(inplace=True)


# In[83]:


df


# ### 4 . Univariate analysis ( single variable Distribution )

# * Histogram for house Price

# In[84]:


pip install --upgrade seaborn 


# In[85]:


import warnings
warnings.filterwarnings("ignore")


# In[86]:


plt.figure(figsize=(10,5))
sns.histplot(df["SalePrice"], kde=True, color="green")

plt.title("Distribution of House Prices")

plt.xlabel("SalePrice")
plt.ylabel("Count")

plt.show()


# * Count plot for categorical variable 

# ### 5 .  Multivariate Analysis ( Relationships Between Variables )

# * Correlation Heatmap

# In[87]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.show()


# * Scatterplot - Price vs Size 

# In[88]:


print(df.columns.tolist())


# In[89]:


if {"GrLivArea", "SalePrice"}.issubset(df.columns):
    tmp = df[['GrLivArea','SalePrice']].dropna()

    plt.figure(figsize=(8,5))
    sns.scatterplot(data=tmp, x="GrLivArea", y="SalePrice")
    plt.title("House Size (GrLivArea) vs Sale Price")
    plt.xlabel("House Size (GrLivArea)")
    plt.ylabel("Sale Price")
    plt.show()
else:
    print("Columns missing! Available:", df.columns.tolist())


# ### 6 .  Feature Engineering 

# * Price per squre foot ( if Size available )

# In[90]:


# Price per Square Foot
df = df[df["GrLivArea"] > 0]


df["Price_per_sqft"] = df["SalePrice"] / df["GrLivArea"]

# Check first rows
print(df[["SalePrice", "GrLivArea", "Price_per_sqft"]].head())


# In[91]:


# House Age
df["HouseAge"] = 2025 - df["YearBuilt"]

df[["SalePrice","GrLivArea","Price_per_sqft","HouseAge"]].head()


# ### 7 . Market Trends (Time Anlysis)

# In[92]:


trend = df.groupby("YrSold")["SalePrice"].mean().reset_index()
plt.figure(figsize=(8,5))

sns.lineplot(data=trend, x="YrSold", y="SalePrice", marker="o")
plt.title("Average Sale Price Over Years")

plt.show()


# ### 8 . Customer Preference & Amenities

# In[94]:


plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="BedroomAbvGr", y="SalePrice")

plt.title("Impact of Bedrooms on Price")

plt.show()


# Through this Exploratory Data Analysis (EDA) on real estate pricing, we discovered valuable insights about the factors influencing house prices.  
# 
# - *Distribution Analysis* revealed that house prices are slightly right-skewed, with most properties concentrated in the mid-price range.  
# - *Correlation Analysis* showed strong relationships between house size (GrLivArea), overall quality, and sale price, making them key predictors.  
# - *Feature Engineering* such as Price per Square Foot and House Age provided deeper understanding of property valuation.  
# - *Market Trends* highlighted fluctuations in average house prices over different years, showing the effect of economic conditions on real estate.  
# - *Customer Preferences* analysis showed that features like number of bedrooms and amenities also impact property value.  
# 
# Overall, this project demonstrates how EDA helps uncover hidden patterns, identify significant features, and prepare the dataset for predictive modeling. These insights can guide better pricing strategies and decision-making in the real estate market.  
# 
# ✨ This analysis not only makes the dataset more understandable but also adds value for future machine learning applications. ✨

# In[ ]:




