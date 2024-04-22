from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import pickle
import base64
from io import BytesIO
import requests
import nltk
import json
from googleapiclient.discovery import build

app = Flask(__name__)

# Load the saved model
model = load_model('food_classification_model.h5')

# Load the saved label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# API credentials
RECIPE_API_KEY = 'PWeTxfMYkqnss6nVnMb4XA==a3sfE77hHqCvudfX'
RECIPE_API_URL = 'https://api.api-ninjas.com/v1/recipe'
NUTRITION_API_KEY = 'Hsb55plrWhUkuWf+ka8I2g==yR4UYOZFC3ZzRAAL'
NUTRITION_API_URL = 'https://api.api-ninjas.com/v1/nutrition?query='

# YouTube API credentials
YOUTUBE_API_KEY = 'AIzaSyAMn3RfzAhlNGeD46RqbvhH-jFtI3ZcjVY'

def predict_class(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    predictions = model.predict(image)
    predicted_class = label_encoder.inverse_transform([np.argmax(predictions[0])])[0]
    return predicted_class

def get_recipe(query):
    headers = {'X-Api-Key': RECIPE_API_KEY}

    # First, search for the original query
    params = {'query': query}
    response = requests.get(RECIPE_API_URL, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if data:
            recipe = data[0]
            instructions = []
            for sentence in nltk.sent_tokenize(recipe['instructions']):
                instructions.append(sentence)
            return instructions

    # If no results found, replace '_' with ' ' and search again
    query_with_spaces = query.replace('_', ' ')
    params = {'query': query_with_spaces}
    response = requests.get(RECIPE_API_URL, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if data:
            recipe = data[0]
            instructions = []
            for sentence in nltk.sent_tokenize(recipe['instructions']):
                instructions.append(sentence)
            return instructions

    # If still no results, search for individual words
    words = query.split('_')
    for i in range(len(words)):
        query = ' '.join(words[:i+1])
        params = {'query': query}
        response = requests.get(RECIPE_API_URL, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if data:
                recipe = data[0]
                instructions = []
                for sentence in nltk.sent_tokenize(recipe['instructions']):
                    instructions.append(sentence)
                return instructions

    return None

def get_nutrition(query):
    headers = {'X-Api-Key': NUTRITION_API_KEY}
    
    # Replace '_' with ' ' in the query
    query = query.replace('_', ' ')
    
    response = requests.get(NUTRITION_API_URL + query, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data:
            calories = data[0]['calories']
            return calories
    
    return None

def get_youtube_videos(query):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    
    # Replace '_' with ' ' in the query and add 'recipe'
    query = f"{query.replace('_', ' ')} recipe"
    
    request = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        maxResults=5
    )
    response = request.execute()
    
    videos = []
    for item in response['items']:
        video_title = item['snippet']['title']
        video_id = item['id']['videoId']
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        thumbnail_url = item['snippet']['thumbnails']['high']['url']
        videos.append({'title': video_title, 'url': video_url, 'thumbnail': thumbnail_url})
    
    return videos

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['image']
        image_bytes = file.read()
        image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        predicted_class = predict_class(image_array)
        
        # Convert the image to base64 for display
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Get recipe instructions
        instructions = get_recipe(predicted_class)
        
        # Get calorie information
        calories = get_nutrition(predicted_class)
        
        # Get YouTube video recommendations
        youtube_videos = get_youtube_videos(predicted_class)
        
        return render_template('result.html', predicted_class=predicted_class, image_data=image_base64, instructions=instructions, calories=calories, youtube_videos=youtube_videos)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)