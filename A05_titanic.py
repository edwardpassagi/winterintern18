import pandas as pd
import numpy as np

pd.set_option('display.height', 10000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 10000)


# read in the data
path = r'C:/Users/edward_p/PycharmProjects/basicDataExploration/input/'
data = pd.read_csv(path + 'train_titanic.csv')

# check the number of rows and columns
print(data.shape)

# have a quick view of data
print(data.head()) # the first 5 rows
print(data.tail()) # the last 5 rows

# check the data types
print(data.dtypes)

# check the missing values
print(data.isnull().sum())

# check the duplicate data
print(data.PassengerId.nunique())


# Data distribution check
# 1. Overall view
print(data.describe())

# 2. View separately
print(data.Embarked.describe())
print(data.Embarked.value_counts(dropna=False))
print(data.Cabin.value_counts())
print(data.Cabin.value_counts().head(20))
print(data.Pclass.value_counts())
print(data.Survived.value_counts())



# Data sorting
data.sort_values('PassengerId', ascending=False, inplace=True)
print(data.head())

# Data duplicate
data.drop_duplicates('PassengerId', keep='first', inplace=True)
data = data.drop_duplicates('PassengerId', keep='first')
print(data.shape)

# fillin missing values
avr_age = int(data['Age'].mean())
avr_age = float(data['Age'].mean())
data['Age'].fillna(avr_age, inplace=True)

# group by
data_gp = data.groupby('Pclass').agg({'PassengerId': 'count',
                                      'Age': 'mean'
                                      })
data_gp.reset_index(inplace=True)
print(data_gp)

data_gp = data.groupby(['Survived', 'Sex']).agg({'PassengerId': 'count',
                                      'Age': 'mean'
                                      })
data_gp.reset_index(inplace=True)
print(data_gp)

data_gp = data.groupby('Sex').agg({'PassengerId': 'count',
                                      'Age': 'mean'
                                      })
data_gp.reset_index(inplace=True)
print(data_gp)