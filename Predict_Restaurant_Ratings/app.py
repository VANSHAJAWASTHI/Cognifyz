import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

df = pd.read_csv('Dataset.csv')

df['Cuisines'].fillna('Unknown', inplace=True)

X = df[['Country Code', 'City', 'Locality', 'Longitude', 'Latitude', 'Cuisines',
       'Has Table booking', 'Has Online delivery', 'Is delivering now', 'Price range', 'Votes']]
y = df['Aggregate rating']

cat_f = ['Country Code', 'City', 'Locality', 'Cuisines', 
         'Has Table booking', 'Has Online delivery', 'Is delivering now']
num_f = ['Longitude', 'Latitude', 'Price range', 'Votes']

prep = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_f),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_f)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate(model, name):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"{name} - MSE: {mse}")
    print(f"{name} - R2: {r2}")
    return mse, r2

lin_model = LinearRegression()
train_and_evaluate(lin_model, 'Linear Regression')

tree_model = DecisionTreeRegressor(random_state=42)
train_and_evaluate(tree_model, 'Decision Tree')

mdl = {
    'RF': RandomForestRegressor(random_state=42),
    'GB': GradientBoostingRegressor(random_state=42)
}

param = {
    'RF': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [10, 20]
    },
    'GB': {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.01, 0.1],
        'model__max_depth': [3, 5]
    }
}

res = {}
for nm, m in mdl.items():
    pipe = Pipeline(steps=[('prep', prep), ('model', m)])
    search = GridSearchCV(pipe, param[nm], cv=5, scoring='r2', n_jobs=-1)
    search.fit(X_train, y_train)
    
    best_m = search.best_estimator_
    pred = best_m.predict(X_test)
    
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    
    res[nm] = {'mse': mse, 'r2': r2, 'best_params': search.best_params_}

for nm, r in res.items():
    print(f"{nm} - MSE: {r['mse']}")
    print(f"{nm} - R2: {r['r2']}")
    print(f"{nm} - Best Params: {r['best_params']}")

best_nm = max(res, key=lambda k: res[k]['r2'])
best_model = mdl[best_nm]
pipe = Pipeline(steps=[('prep', prep), ('model', best_model)])
pipe.fit(X_train, y_train)
joblib.dump(pipe, 'best_model.pkl')

if hasattr(pipe.named_steps['model'], 'feature_importances_'):
    fi = pipe.named_steps['model'].feature_importances_
    fn = (pipe.named_steps['prep']
          .transformers_[1][1]
          .get_feature_names_out(cat_f))
    all_fn = num_f + list(fn)
    imp_f = pd.Series(fi, index=all_fn).sort_values(ascending=False)
    print(f"Feature Importances for {best_nm}:")
    print(imp_f)

missing_data = df.isnull().sum()
print("Missing Data Summary:")
print(missing_data)

data_types = df.dtypes
print("Data Types Summary:")
print(data_types)
