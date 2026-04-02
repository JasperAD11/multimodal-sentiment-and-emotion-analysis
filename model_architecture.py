import tensorflow as tf
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import model as m

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers, models, initializers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import TextVectorization, Input, Embedding, LSTM, Dropout, Dense
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_ensemble_model(sentiment_model_path, emotion_model_path):
    # Load the models
    sentiment_model = load_model(sentiment_model_path)
    emotion_model = load_model(emotion_model_path)

    # Freeze the models to prevent training
    sentiment_model.trainable = False
    emotion_model.trainable = False

    # Define new input layers
    sentiment_input = Input(shape=sentiment_model.input_shape[1:], name="sentiment_input")
    emotion_input = Input(shape=emotion_model.input_shape[1:], name="emotion_input")

    # Pass the inputs through the respective models
    sentiment_output = sentiment_model(sentiment_input)
    emotion_output = emotion_model(emotion_input)

    # Create the joint model
    joint_model = Model(
        inputs=[sentiment_input, emotion_input],
        outputs=[sentiment_output, emotion_output]
    )

    return joint_model

def predict_ensemble_model(model, texts, vectorizer, max_length=300, neutral_threshold=0.3, emotion_threshold=0.15):
    # Tokenize and pad the input texts
    input = vectorizer(texts)

    # Make predictions with the joint model
    predictions = model.predict({
        'sentiment_input': input,
        'emotion_input': input
    })

    # Get the sentiment prediction
    sentiment_prediction = predictions[0]

    # Convert sentiment prediction to 'positive' or 'negative' based on threshold of 0.5
    sentiment_label = "positive" if sentiment_prediction[0] > 0.5 else "negative"

    # Get emotion predictions
    emotion_predictions = predictions[1]

    # Define emotion labels (adjust to your actual labels)
    emotion_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
                      'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
                      'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
                      'remorse', 'sadness', 'surprise', 'neutral']

    # Map the emotion predictions to the emotion labels
    emotion_results = {emotion_labels[i]: emotion_predictions[0][i] for i in range(len(emotion_labels))}

    # Check if 'neutral' emotion has score > neutral_threshold
    if emotion_results.get('neutral', 0) >= neutral_threshold:
        # If neutral is above the threshold, only return "neutral"
        return {
            'sentiment': sentiment_label,
            'emotion': ['neutral']
        }

    # Filter emotions: return all emotions > emotion_threshold, excluding 'neutral'
    filtered_emotions = {emotion: score for emotion, score in emotion_results.items() if score > emotion_threshold and emotion != 'neutral'}

    # If no emotions are above the threshold, return only the emotion with the highest score, excluding 'neutral'
    if not filtered_emotions:
        max_emotion = max((emotion_results[key], key) for key in emotion_results if key != 'neutral')
        filtered_emotions = {max_emotion[1]: max_emotion[0]}

    # Return the predictions
    return {
        'sentiment': sentiment_label,  # Sentiment prediction as 'positive' or 'negative'
        'emotion': list(filtered_emotions.keys())  # List of emotions above threshold or best emotion
    }
