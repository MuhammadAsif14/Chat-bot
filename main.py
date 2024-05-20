from flask import Flask, render_template, request
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

from training import intents, ignoreLetters

lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

model = tf.keras.models.load_model('chatbot_model.h5')

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))





def clean_message(message):
    message_words = nltk.word_tokenize(message)
    message_words = [lemmatizer.lemmatize(word.lower()) for word in message_words if word not in ignoreLetters]
    return message_words


def bag_of_words(message):
    message_words = clean_message(message)
    bag = [0] * len(words)
    for w in message_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(message, model):
    bow = bag_of_words(message)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)