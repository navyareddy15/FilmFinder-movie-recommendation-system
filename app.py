from flask import Flask, render_template, request, jsonify
import pandas as pd
import difflib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
movies_data = pd.read_csv('main.csv')

# Selecting the relevant features for recommendation
selected_features = ["genres", "actor_1_name", "actor_2_name", "director_name"]

# Replacing null values
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine all the selected features
movies_data['combined_features'] = movies_data["genres"] + " " + movies_data["actor_1_name"] + " " + movies_data["actor_2_name"] + " " + movies_data["director_name"]

# Vectorize combined features
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(movies_data['combined_features'])

# Compute cosine similarity
similarity = cosine_similarity(feature_vectors)
def recommend_movies(movie_name, top_n=5):
    # Finding the closest match with input
    find_close_match = difflib.get_close_matches(movie_name, movies_data["movie_title"].tolist())
    closest_match = find_close_match[0] if find_close_match else None
    
    if closest_match:
        # Finding index in dataset
        index_of_movie = movies_data[movies_data.movie_title == closest_match].index[0]
        
        # Similarity list
        similarity_score = list(enumerate(similarity[index_of_movie]))
        
        # Sort this list to have the highest similarity score
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        
        # Get top recommended movies
        recommended_movies = []
        for i, movie in enumerate(sorted_similar_movies):
            index = movie[0]
            title = movies_data.loc[index, "movie_title"]
            genres = movies_data.loc[index, "genres"]
            actor_1_name = movies_data.loc[index, "actor_1_name"]
            actor_2_name = movies_data.loc[index, "actor_2_name"]
            
            recommended_movies.append({
                "title": title,
                "genres": genres,
                "actor_1_name": actor_1_name,
                "actor_2_name": actor_2_name,
            })

            if i >= top_n:
                break

        return recommended_movies
    else:
        return [{"title": "Movie not found in the dataset."}]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    movie_name = request.form['movie_name']
    recommended_movies = recommend_movies(movie_name)
    return jsonify(recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
