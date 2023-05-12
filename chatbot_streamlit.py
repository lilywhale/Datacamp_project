import streamlit as st
from streamlit_chat import message as st_message

import random
import json
import nltk
nltk.download('punkt')
import torch
import numpy
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#If you want to use the original dataset change the name to intents_final1.json
with open('intents_corrected.json', 'r') as json_data:
    intents = json.load(json_data)

# Here to use the original data you need to use data1.pth
FILE = "data_corrected.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

if "history" not in st.session_state:
    st.session_state.history = []


st.title("Customer Service Chatbot")

st.write(" Here are a few example of things you can ask to the customer support bot")
st.write("  - When are you going to add the rick and morty season 3")
st.write("  - Why is the update making my phone crash everyday ? ")
st.write("  - If I purchase a flight with miles will i get a discount")
st.write("  - How can I track my order")
st.write("  - Are you guys looking into the problem ?")

st.write("You can also tell the bot about your customer experience.")
st.caption("Tip to delete your last query faster ctrl+A and delete ;-)")


def generate_answer():
    sentence = st.session_state.input_text
   
    st.session_state.history.append({"message": sentence, "is_user": True})
    
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    predicted = torch.max(output, dim=1)[1]


    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                st.session_state.history.append({"message": str(response), "is_user": False})
    else:
        st.session_state.history.append({"message": "I do not understand...", "is_user": False})




st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)

for i, chat in enumerate(st.session_state.history):
    st_message(**chat, key=str(i)) #unpacking
