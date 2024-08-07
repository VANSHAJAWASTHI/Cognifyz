import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer

df = pd.read_csv("Dataset.csv")

df['Cuisines'].fillna('Unknown', inplace=True)

ft = df[['Restaurant ID', 'Country Code', 'City', 'Locality', 'Cuisines', 'Price range', 'Aggregate rating', 'Votes']]

cat_feats = ['Country Code', 'City', 'Locality', 'Cuisines', 'Price range']
num_feats = ['Aggregate rating', 'Votes']

prep = ColumnTransformer(
    transformers=[
        ('c', OneHotEncoder(handle_unknown='ignore'), cat_feats),
        ('n', StandardScaler(), num_feats)
    ])

pf = prep.fit_transform(ft)

def rec_rest(user_pref, top_n=5):
    user_df = pd.DataFrame([user_pref], columns=cat_feats + num_feats)
    user_enc = prep.transform(user_df)
    sim = cosine_similarity(user_enc, pf)
    idx = sim[0].argsort()[-top_n:][::-1]
    return df.iloc[idx][['Restaurant Name', 'Cuisines', 'Price range', 'Aggregate rating', 'Votes', 'City', 'Locality']]

up = ['Makati City', 'Century City Mall, Poblacion, Makati City', 'Japanese', '3', 4.5, 300]

rec = rec_rest(up)
print("Recommended Restaurants:")
print(rec)
