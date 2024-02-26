import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
import speech_recognition as sr
import pyttsx3
import streamlit as st

nltk.download('popular', quiet=True)
nltk.download('punkt')
nltk.download('wordnet')

warnings.filterwarnings('ignore')

# Reading in the corpus
with open('chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# Tokenization
sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating response
def response(user_response):
    buddy_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        buddy_response = buddy_response + "I am sorry! I don't understand you"
        return buddy_response
    else:
        buddy_response = buddy_response + sent_tokens[idx]
        return buddy_response


def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("BUDDY: Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
        try:
            user_response = recognizer.recognize_google(audio)
            print("You:", user_response)
            return user_response.lower()
        except sr.UnknownValueError:
            return ""


def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def chat():
    flag = True
    st.write("BUDDY: My name is Buddy. I will answer your queries about Chatbots. If you want to exit, type Bye!")
    while (flag == True):
        user_response = speech_to_text()
        if (user_response != 'quit'):
            if (user_response == 'thanks' or user_response == 'thank you'):
                flag = False
                text_to_speech("You are welcome..")
            else:
                if (greeting(user_response) != None):
                    text_to_speech("BUDDY: " + greeting(user_response))
                else:
                    buddy_response = response(user_response)
                    st.write("BUDDY:", buddy_response)
                    text_to_speech("BUDDY: " + buddy_response)
                    sent_tokens.remove(user_response)
        else:
            flag = False
            text_to_speech("Bye! take care..")


chat()
