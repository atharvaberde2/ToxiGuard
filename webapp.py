# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:30:02 2025

@author: athar
"""

import streamlit as st
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification, TFBertModel
import tensorflow as tf
from lime.lime_text import LimeTextExplainer

from huggingface_hub import login, upload_folder
import openai
import os

# Log in (only needed once)
login(token="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") 




st.markdown(
    """
    <style>
    .stApp {
        background-color: #1D1E22;
        color: white;
    }
    .stButton>button {
        background-color: #E63946;
        color: white;
        border-radius: 8px;
        font-size: 16px;
    }
    .stTextInput>div>div>input {
        background-color: #2C2F33;
        color: white;
    }
    .stMarkdown {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# Title
st.title("🛑 Hate Speech Detector")

# User Input
text_input = st.text_area("Enter text:", placeholder="Type here...")
tokenizer1 = BertTokenizer.from_pretrained("bert-base-uncased")
model1 = TFBertForSequenceClassification.from_pretrained(
    "Atharva1244/hate_speech_classifier", 
    ignore_mismatched_sizes=True
)

# Submit Button

if st.button('Predict'):
    if text_input:
        # Tokenize the input text
        inputs = tokenizer1(text_input, return_tensors="tf", padding=True, truncation=True, max_length=512)

        # Get model predictions
        outputs = model1(**inputs)

        # Extract logits and convert to probabilities
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=-1)

        # Get the predicted class
        predicted_class = tf.argmax(probabilities, axis=-1).numpy()

        # Display the result
        if predicted_class[0] == 2:
            st.write('Your comment was not offensive nor hateful.')
        elif predicted_class[0] == 1:
            st.write('Your comment was offensive. It uses offensive language.')
            user_input = 'Identify the offensive phrases. Rewrite this to make it less offensive'
           
        else:
            st.write('Your comment was very hateful.')
            user_input = 'Identify the hateful words/phrases. Rewrite this to make it less hateful'
           
        #st.write(f"Predicted Class: {predicted_class[0]}")
        #st.write(f"Probabilities: {probabilities.numpy()}")
        
    else:
        st.warning("Please enter some text to analyze.")
