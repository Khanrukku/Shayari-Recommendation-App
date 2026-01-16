from flask import Flask, render_template, request, jsonify
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

app = Flask(__name__)

# Sample Shayari database (in production, this would be from MongoDB)
SHAYARI_DATABASE = [
    {
        'id': 1,
        'text': 'Dil ki gehraaiyon mein chhupa hai ek raaz, Mohabbat ka wo safar jo kabhi na ho paaya shuru',
        'emotion': 'sad',
        'theme': 'love',
        'author': 'Unknown'
    },
    {
        'id': 2,
        'text': 'Khushiyon ke phool khilte hain, Jab dil mein umeed ki kiran jagti hai',
        'emotion': 'happy',
        'theme': 'motivation',
        'author': 'Unknown'
    },
    {
        'id': 3,
        'text': 'Teri yaad mein raat guzar jaati hai, Subah hoti hai par tera intezaar rahta hai',
        'emotion': 'romantic',
        'theme': 'love',
        'author': 'Unknown'
    },
    {
        'id': 4,
        'text': 'Mushkil raahon mein bhi hausla rakho, Manzil zaroor milegi ek din',
        'emotion': 'motivational',
        'theme': 'inspiration',
        'author': 'Unknown'
    },
    {
        'id': 5,
        'text': 'Dosti wo rishta hai jo kabhi nahi tootta, Dil se dil ka connection hota hai',
        'emotion': 'friendship',
        'theme': 'friendship',
        'author': 'Unknown'
    }
]

class ShayariRecommender:
    """
    Content-based recommendation system for Shayari using NLP techniques.
    Uses TF-IDF vectorization and cosine similarity for recommendations.
    """
    
    def __init__(self, shayari_data):
        self.shayari_df = pd.DataFrame(shayari_data)
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.tfidf_matrix = None
        self._build_model()
    
    def _build_model(self):
        """Build TF-IDF matrix for all shayari"""
        self.tfidf_matrix = self.vectorizer.fit_transform(self.shayari_df['text'])
    
    def analyze_emotion(self, text):
        """
        Analyze emotion from user input using TextBlob sentiment analysis.
        Returns emotion category based on polarity and subjectivity.
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.5:
            return 'happy'
        elif polarity < -0.3:
            return 'sad'
        elif polarity > 0.1:
            return 'romantic'
        elif polarity < 0:
            return 'motivational'
        else:
            return 'friendship'
    
    def recommend_by_emotion(self, emotion, top_n=5):
        """
        Recommend shayari based on emotion category.
        Returns top N shayari matching the emotion.
        """
        filtered = self.shayari_df[self.shayari_df['emotion'] == emotion]
        
        if len(filtered) == 0:
            # If no exact match, return most similar
            return self.shayari_df.head(top_n).to_dict('records')
        
        return filtered.head(top_n).to_dict('records')
    
    def recommend_by_similarity(self, user_text, top_n=5):
        """
        Recommend shayari based on content similarity using cosine similarity.
        Uses TF-IDF vectors for semantic matching.
        """
        user_vector = self.vectorizer.transform([user_text])
        similarity_scores = cosine_similarity(user_vector, self.tfidf_matrix)[0]
        
        # Get top N indices
        top_indices = np.argsort(similarity_scores)[-top_n:][::-1]
        
        recommendations = []
        for idx in top_indices:
            shayari = self.shayari_df.iloc[idx].to_dict()
            shayari['similarity_score'] = float(similarity_scores[idx])
            recommendations.append(shayari)
        
        return recommendations

# Initialize recommender
recommender = ShayariRecommender(SHAYARI_DATABASE)

@app.route('/')
def home():
    """Home page route"""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    API endpoint for getting recommendations.
    Accepts user input and returns recommended shayari.
    """
    data = request.get_json()
    user_input = data.get('text', '')
    method = data.get('method', 'emotion')  # 'emotion' or 'similarity'
    
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    if method == 'emotion':
        # Emotion-based recommendation
        detected_emotion = recommender.analyze_emotion(user_input)
        recommendations = recommender.recommend_by_emotion(detected_emotion)
        
        return jsonify({
            'detected_emotion': detected_emotion,
            'recommendations': recommendations,
            'method': 'emotion-based'
        })
    
    else:
        # Similarity-based recommendation
        recommendations = recommender.recommend_by_similarity(user_input)
        
        return jsonify({
            'recommendations': recommendations,
            'method': 'similarity-based'
        })

@app.route('/api/emotions')
def get_emotions():
    """Get list of available emotion categories"""
    emotions = self.shayari_df['emotion'].unique().tolist()
    return jsonify({'emotions': emotions})

if __name__ == '__main__':
    print("Starting Shayari Recommendation System...")
    print("Access at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
