import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('/Users/digreelal/Docs/House Price Prediction Project/Dataset/housing 2.csv')
print(data.head())
print(data.tail())
print(data.info())
print(data['ocean_proximity'].value_counts())
print(data.describe())
# import matplotlib.pyplot as plt
# data.hist(bins=50, figsize=(12, 8))
# plt.show()


# Creating a test set and training set
def shuffle_and_split(data, test_ratio):
    np.random.seed(42) # Set the seed for reproducibility
    shuffled_indecies = np.random.permutation(len(data)) # This return shuffled indecies
    test_set_size = int(len(data) * test_ratio)
    test_indecies = shuffled_indecies[:test_set_size]
    train_indecies = shuffled_indecies[test_set_size:]
    return data.iloc[train_indecies], data.iloc[test_indecies]


train, test = shuffle_and_split(data, 0.2)
print(train)
print(test)

# Create a new column for income category to stratify the data based on income
data['income_cat'] = pd.cut(data['median_income'], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1,2,3,4,5])

# import matplotlib.pyplot as plt
# data["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)  
# plt.title("Income Categories Distribution") 
# plt.xlabel("Income Category")
# plt.ylabel("Number of Instances")
# plt.show()


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split (data, data['income_cat']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

print(strat_train_set)
print(strat_test_set)

# import matplotlib.pyplot as plt
# strat_train_set["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)  
# plt.title("Income Categories Distribution") 
# plt.xlabel("Income Category")
# plt.ylabel("Number of Instances")
# plt.show()


# import matplotlib.pyplot as plt
# strat_test_set["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)  
# plt.title("Income Categories Distribution") 
# plt.xlabel("Income Category")
# plt.ylabel("Number of Instances")
# plt.show()


# Visualizing the data
# Lets remove the income_cat column
for sett in (strat_train_set, strat_test_set):
    sett.drop('income_cat', axis = 1, inplace=True)

data = strat_train_set.copy()

# data.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
# plt.show()

# data.plot(kind="scatter", x="longitude", y="latitude", grid=True, cmap='jet', c='median_house_value')
# plt.show()
# data.drop('ocean_proximity', axis = 1, inplace=True)
# corr_matrix = data.corr()
# print(corr_matrix)

from pandas.plotting import scatter_matrix

# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(data[attributes], figsize=(12, 8))
# plt.show()

# Preprocessing the data
housing = strat_train_set.drop('median_house_value', axis = 1)
housing_labels = strat_train_set['median_house_value'].copy()

print(housing)
print(housing_labels)

# Preprocessing on Numerical Attributes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')

housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)


print(imputer.statistics_)

X = imputer.transform(housing_num)
print(X)


# Preprocessing on Categorical Attributes

housing = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
housing['ocean_proximity'] = data['ocean_proximity']
housing = housing[['ocean_proximity']]
print(housing)

print(set(housing['ocean_proximity'])) 

# Ordinal Encoder
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

housing_cat = ordinal_encoder.fit_transform(housing)
housing_cat = pd.DataFrame(housing_cat, columns=housing.columns, index=housing.index)
print(housing_cat)

# One Hot Encoder
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder()

housing_cat = one_hot_encoder.fit_transform(housing)
housing_cat = pd.DataFrame(housing_cat.toarray(), columns=['<1H OCEAN', 'ISLAND', 'INLAND', 'NEAR BAY', 'NEAR OCEAN'], index=housing.index)
print(housing_cat)


data = pd.concat([data, housing_cat], axis = 1)
print(data)

data = data.drop('ocean_proximity', axis=1)
print(data) 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)
print(data_scaled)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)
print(data_scaled)


housing = strat_train_set.drop('median_house_value', axis = 1)
housing_labels = strat_train_set['median_house_value'].copy()

print(housing)
print(housing_labels)

housing = housing.drop('ocean_proximity', axis = 1)
print(housing)


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

mypipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('standardize', StandardScaler())
])

pl = mypipeline.fit_transform(housing)
pl = pd.DataFrame(pl, columns=housing.columns, index=housing.index)
print(pl)





