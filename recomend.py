import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-item ratings matrix
data = {
    'User': ['User1', 'User2', 'User3', 'User4', 'User5'],
    'ItemA': [5, 4, 1, np.nan, 3],
    'ItemB': [4, np.nan, 2, 4, 3],
    'ItemC': [1, 3, 4, 2, 5],
    'ItemD': [np.nan, 4, 5, 3, 2],
    'ItemE': [2, 3, np.nan, 5, 4],
}

df = pd.DataFrame(data)

# Fill NaN values with 0 for cosine similarity calculation
df_filled = df.fillna(0)

# Create user-user similarity matrix
user_similarity = cosine_similarity(df_filled.drop('User', axis=1))
user_similarity_df = pd.DataFrame(user_similarity, index=df['User'], columns=df['User'])

# Function to get recommendations for a user
def get_recommendations(user, n=2):
    similar_users = user_similarity_df[user].sort_values(ascending=False)[1:]
    recommendations = pd.Series(dtype='float64')
    
    for similar_user, similarity in similar_users.items():
        user_ratings = df[df['User'] == similar_user].drop('User', axis=1)
        user_ratings = user_ratings.dropna(axis=1)
        
        for item in user_ratings.columns:
            if pd.isna(df[df['User'] == user][item].values[0]):
                if item not in recommendations:
                    recommendations[item] = 0
                recommendations[item] += similarity * user_ratings[item].values[0]
    
    return recommendations.sort_values(ascending=False).head(n)

# Get recommendations for a specific user
user_to_recommend = 'User1'
recommendations = get_recommendations(user_to_recommend)
print(f"Recommendations for {user_to_recommend}:\n{recommendations}")
