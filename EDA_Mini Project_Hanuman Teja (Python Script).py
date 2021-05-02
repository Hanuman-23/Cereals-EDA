# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 19:04:07 2021

@author: DELL
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib as mpl


mpl.style.use('ggplot')
%matplotlib inline

cereals=pd.read_csv('C:/Users/DELL/Downloads/cereals_data.csv')
cereals.info()
df=pd.DataFrame(cereals)
df
df.isnull().sum()

#FILLING MISSING VALUES
cateogry_columns=df.select_dtypes(include=['object']).columns.tolist()
integer_columns=df.select_dtypes(include=['int64','float64']).columns.tolist()

for column in df:
    if df[column].isnull().any():
        if(column in cateogry_columns):
            df[column]=df[column].fillna(df[column].mode()[0])
        else:
            df[column]=df[column].fillna(df[column].mean)

#We have 2 missing values in 'potass', 1 in 'carbo' and 1 in 'sugars'
#We are plotting a histograms for 'potass','carbo' and 'sugars' to know whether 
#the curve is symmentric or not
plt.hist(cereals.potass,bins='auto',facecolor = 'green')
plt.xlabel('potass')
plt.ylabel('counts')
plt.title('Histogram of Potass')

plt.hist(cereals.carbo,bins='auto',facecolor = 'green')
plt.xlabel('carbo')
plt.ylabel('counts')
plt.title('Histogram of Carbo')

plt.hist(cereals.sugars,bins='auto',facecolor = 'green')
plt.xlabel('sugars')
plt.ylabel('counts')
plt.title('Histogram of Sugars')

#After plotting Histograms for 'potass','carbo' and 'sugars', we got to know the curves
#are not symmentric. Hence, we are replacing the missing values with the 'medians'

#Calculating medians for 'potass','carbo' and 'sugars'
cereals.potass.median()
cereals.carbo.median()
cereals.sugars.median()
#Replacing missing values with MODE for categorical Variable
mode_value=hp['MasVnrType'].mode()
hp['MasVnrType'].fillna(mode_value[0],inplace=True)  

#Replacing missing values in 'Potass' with median
cereals['potass'].fillna(cereals['potass'].median(),inplace=True)
cereals.potass.isnull().sum()
#Replacing missing values in 'carbo' with median
cereals['carbo'].fillna(cereals['carbo'].median(),inplace=True)
cereals.carbo.isnull().sum()
#Replacing missing values in 'sugars' with median
cereals['sugars'].fillna(cereals['sugars'].median(),inplace=True)
cereals.carbo.isnull().sum()

#checking whether the missing values are imputed
cereals.isnull().sum()
cereals.info()

#Let us change the 'mfr' and 'type' to Categorical

cate=['mfr', 'type']

for c in cate:
    cereals[cate]=cereals[cate].astype('category')
cereals[cate].info()

#Finding unique values in Categorical Variables
cereals.mfr.unique().shape
cereals.mfr.unique()

cereals.type.unique().shape
cereals.type.unique()

# 'mfr' vs 'rating (Ploting Bar Plot to analyse Brand Rating)
import seaborn as sns
plt.figure(figsize=(10,10))
plt.title('Manufacturer Rating')
sns.barplot(x='mfr', y='rating', hue='shelf',data=cereals)

# 'cereals' vs 'rating (Ploting Bar Plot to analyse cereals Rating)
import seaborn as sns
plt.figure(figsize=(10,10))
plt.title('cereals Rating')
sns.barplot(x='rating', y='name', hue='type',data=cereals)

# 'mfr' vs 'name' (Number of products by manufacturer)
cereals.name.groupby(cereals.mfr).describe()
cereals[cereals.mfr=='A']
cereals[cereals.mfr=='G']
cereals[cereals.mfr=='K']
cereals[cereals.mfr=='N']
cereals[cereals.mfr=='P']
cereals[cereals.mfr=='Q']
cereals[cereals.mfr=='R']

#This gives the count of cereals supplied by the mfr
cereals_by_mfr=cereals.mfr.value_counts()
cereals_by_mfr
pd.crosstab(cereals.mfr,cereals.type,rownames=['mfr'],colnames=['type'])

#Plotting Count plot for 'mfr' grouped by 'type'
sns.countplot(x='mfr', hue='type',data = cereals)

#plotting Histogram for all the variables
cereals.hist()
cereals.boxplot()

#plotting Boxplot to know the distribution of calories and outliers
plt.boxplot(cereals['calories'])
plt.title('Distribution of calories')
plt.xlabel('Calories')

calories=pd.Series(np.array(cereals.calories),index=[cereals.name])
#High Calorie cereals
High_calories=calories[calories>120]
print(High_calories.sort_values(ascending=False))
#Low calorie cereals
Low_calories=calories[calories<90]
print(Low_calories.sort_values(ascending=False))

cereals.protein.describe()
cereals.fat.describe()
cereals.sodium.describe()
cereals.fiber.describe()
cereals.carbo.describe()
cereals.sugars.describe()
cereals.potass.describe()
cereals.vitamins.describe()

#Plotting Scatter plot to know the relation between 'calories' and 'fat'
cereals.type.value_counts
fig,ax=plt.subplots()
colors={'C':'blue','H':'red'}
grouped=cereals.groupby('type')
for key, group in grouped:
    group.plot(ax=ax,kind='scatter',x='calories',y='fat',label=key,color=colors[key])
plt.xlabel('Calories')
plt.ylabel('Fat')
plt.title('Calories vs Fat')
plt.show()

#Plotting regression plot to know the relation between 'calories' and 'fat' with
#regression line
sns.regplot(x='calories', y='fat',data=cereals)
plt.title('Calories vs Fat-Reg plot')

sns.regplot(x='calories', y='protein',data=cereals)
plt.title('Calories vs protein-Reg plot')

sns.regplot(x='calories', y='sodium',data=cereals)
plt.title('Calories vs sodium-Reg plot')

sns.regplot(x='calories', y='fiber',data=cereals)
plt.title('Calories vs fiber-Reg plot')

sns.regplot(x='calories', y='carbo',data=cereals)
plt.title('Calories vs carbohydrates-Reg plot')

sns.regplot(x='calories', y='sugars',data=cereals,color=('magenta'))
plt.title('Calories vs sugars-Reg plot')

sns.regplot(x='calories', y='potass',data=cereals)
plt.title('Calories vs potassium-Reg plot')

sns.regplot(x='calories', y='vitamins',data=cereals)
plt.title('Calories vs vitamins-Reg plot')

#plotting Boxplot to know the distribution of vitamins and outliers
plt.boxplot(cereals['vitamins'])
plt.title('Distribution of vitamins')
plt.xlabel('vitamins')

vitamins=pd.Series(np.array(cereals.vitamins),index=[cereals.name,cereals.mfr])
#High vitamins cereals
High_vitamins=vitamins[vitamins==100]
print(High_vitamins.sort_values(ascending=False))
#Zero vitamins cereals
Zero_vitamins=vitamins[vitamins==0]
print(Zero_vitamins.sort_values(ascending=False))

#Plotting pair plot between all nutrients

da=cereals[['protein','carbo','potass','type']]
sns.pairplot(da,hue='type',kind='reg',palette='spring_r')

#Plotting pair plot
da=cereals[['protein','vitamins','fat','type']]
sns.pairplot(da,hue='type',kind='reg',palette='spring_r')

da=cereals[['protein','sugars','fiber','type']]
sns.pairplot(da,hue='type',kind='reg',palette='spring_r')

sns.regplot(x='protein', y='sodium',data=cereals)
plt.title('Calories vs vitamins-Reg plot)

da=cereals[['protein','sodium','fiber','type']]
sns.pairplot(da,hue='type',kind='reg',palette='spring_r')

da=cereals[['carbo','sodium','sugars','type']]
sns.pairplot(da,hue='type',kind='reg',palette='spring_r')

da=cereals[['potass','sodium','vitamins','type']]
sns.pairplot(da,hue='type',kind='reg',palette='spring_r')

da=cereals[['potass','sugars','vitamins','type']]
sns.pairplot(da,hue='type',kind='reg',palette='spring_r')

da=cereals[['carbo','sugars','vitamins','type']]
sns.pairplot(da,hue='type',kind='reg',palette='spring_r')

da=cereals[['potass','carbo','fiber','type']]
sns.pairplot(da,hue='type',kind='reg',palette='spring_r')

da=cereals[['fat','fiber','vitamins','type']]
sns.pairplot(da,hue='type',kind='reg',palette='spring_r')

da=cereals[['fat','sodium','carbo','type']]
sns.pairplot(da,hue='type',kind='reg',palette='spring_r')

da=cereals[['potass','fat','sugars','type']]
sns.pairplot(da,hue='type',kind='reg',palette='spring_r')


plt.boxplot(cereals['protein'])
plt.title('Distribution of vitamins')
plt.xlabel('protein')

protein=pd.Series(np.array(cereals.protein),index=[cereals.name,cereals.mfr])
#High protein cereals
High_protein=protein[protein>4]
print(High_protein.sort_values(ascending=False))

plt.boxplot(cereals['fiber'])
plt.title('Distribution of fiber')
plt.xlabel('fiber')

fiber=pd.Series(np.array(cereals.fiber),index=[cereals.name,cereals.mfr])
#High fiber cereals
High_fiber=fiber[fiber>6]
print(High_fiber.sort_values(ascending=False))

plt.boxplot(cereals['rating'])
plt.title('Distribution of rating')
plt.xlabel('rating')

rating=pd.Series(np.array(cereals.rating),index=[cereals.name,cereals.mfr,
cereals.shelf,cereals.type])
#High rated cereals
High_rated=rating[rating>60]
print(High_rated.sort_values(ascending=False))

#correlation matrix
data=cereals
df=pd.DataFrame(data)
print(df)
df.corr()
corrMatrix=df.corr()
print(corrMatrix)
sns.heatmap(corrMatrix,annot=True)
plt.show()

import seaborn as sns
plt.figure(figsize=(10,10))
plt.title('Shelf Rating')
sns.barplot(x='shelf', y='rating',data=cereals)

plt.hist(cereals['calories'])
cereals.rating.describe()
