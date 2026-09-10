import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score


# 1. Load the dataset
housing = pd.read_csv('/Users/digreelal/Docs/Machine Learning Official Practice/Dataset/housing 2.csv')
print(housing.head())


# 2. Create a stratified test set
housing['income_cat'] = pd.cut(housing['median_income'],
                               bins = [0, 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels=[1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop('income_cat', axis=1)
    strat_test_set = housing.loc[test_index].drop('income_cat', axis=1)


# We will work on the copy of training data
housing = strat_train_set.copy()

# 3. Separate features and labels
housing_labels = housing['median_house_value'].copy()
housing = housing.drop('median_house_value', axis=1)

print(housing, housing_labels)


# 4. Seperate numerical and categorical columns
num_attributes = housing.drop('ocean_proximity', axis = 1).columns.tolist()
cat_attributes = ['ocean_proximity']



# 5. Lets make the pipeline

# for numerical columns
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])



# for categorical columns
cat_pipeline = Pipeline([
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])


# Construct the full pipeline
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attributes),
    ('cat', cat_pipeline, cat_attributes)
])


# 6. Transformed the data
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)
print(housing_prepared.shape)


# 7. Train the model

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
# lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
# print(f"The root mean squared error for the Linear Regression is {lin_rmse}")
lin_rmse = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(lin_rmse).describe())



# Decision Tree
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared, housing_labels)
dec_preds = dec_reg.predict(housing_prepared)
# dec_rmse = root_mean_squared_error(housing_labels, dec_preds)
# print(f"The root mean squared error for the Decision Tree is {dec_rmse}")
dec_rmse = -cross_val_score(dec_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(dec_rmse).describe())


# Random Forest Regression
rf_reg = RandomForestRegressor()
rf_reg.fit(housing_prepared, housing_labels)
rf_preds = rf_reg.predict(housing_prepared)
# rf_rmse = root_mean_squared_error(housing_labels, rf_preds)
# print(f"The root mean squared error for the Random forest Regression is {rf_rmse}")
rf_rmse = -cross_val_score(rf_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(rf_rmse).describe())