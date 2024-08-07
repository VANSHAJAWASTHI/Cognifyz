import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, f1_score

df = pd.read_csv('Dataset.csv')

df['Cuisines'].fillna('Unknown', inplace=True)
le = LabelEncoder()
df['Cuisines'] = le.fit_transform(df['Cuisines'])

X = df[['Country Code', 'City', 'Locality', 'Longitude', 'Latitude', 'Price range', 'Aggregate rating', 'Votes']]
y = df['Cuisines']

cat_features = ['Country Code', 'City', 'Locality', 'Price range']
num_features = ['Longitude', 'Latitude', 'Aggregate rating', 'Votes']

preproc = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', StandardScaler(), num_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'LogReg': LogisticRegression(max_iter=1000, random_state=42),
    'RandFor': RandomForestClassifier(random_state=42),
    'GradBoost': GradientBoostingClassifier(random_state=42)
}

params = {
    'LogReg': {'model__C': [0.1, 1, 10]},
    'RandFor': {'model__n_estimators': [100, 200], 'model__max_depth': [10, 20]},
    'GradBoost': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.01, 0.1], 'model__max_depth': [3, 5]}
}

res = []

for name, mdl in models.items():
    pipe = Pipeline(steps=[('preproc', preproc), ('model', mdl)])
    search = GridSearchCV(pipe, params[name], cv=5, scoring='accuracy', n_jobs=-1)
    search.fit(X_train, y_train)
    
    best_mdl = search.best_estimator_
    y_pred = best_mdl.predict(X_test)
    
    res.append({
        'model': name,
        'acc': accuracy_score(y_test, y_pred),
        'prec': precision_score(y_test, y_pred, average='weighted'),
        'rec': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'params': search.best_params_,
        'report': classification_report(y_test, y_pred, target_names=le.classes_),
        'conf_matrix': confusion_matrix(y_test, y_pred)
    })

res_df = pd.DataFrame(res)

for _, r in res_df.iterrows():
    print(f"{r['model']} - Accuracy: {r['acc']}")
    print(f"{r['model']} - Precision: {r['prec']}")
    print(f"{r['model']} - Recall: {r['rec']}")
    print(f"{r['model']} - F1 Score: {r['f1']}")
    print(f"{r['model']} - Best Params: {r['params']}")
    print(f"{r['model']} - Classification Report:\n{r['report']}")

best_name = res_df.loc[res_df['f1'].idxmax(), 'model']
best_conf_matrix = res_df.loc[res_df['model'] == best_name, 'conf_matrix'].values[0]

plt.figure(figsize=(10, 7))
sns.heatmap(best_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix for {best_name}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(f'conf_matrix_{best_name}.png')
plt.show()
