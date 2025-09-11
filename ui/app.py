import os
import time

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Fashion Recommendation Chatbot",
    layout="centered"
)

# Custom CSS for Fancy UI with Padding & Avatars
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
    }

    .chat-container {
        max-width: 800px;
        margin: auto;
        padding: 20px;
        border-radius: 12px;
    }

    .user-message {
        background-color: #007bff;
        color: white;
        padding: 12px;
        border-radius: 10px;
        max-width: 75%;
        text-align: right;
        float: right;
        clear: both;
        margin: 10px 0;
    }

    .bot-message {
        background-color: #f1f1f1;
        color: black;
        padding: 12px;
        border-radius: 10px;
        max-width: 75%;
        text-align: left;
        float: left;
        clear: both;
        margin: 10px 0;
    }

    .chat-wrapper {
        padding: 20px 100px;
    }

    .avatar {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        vertical-align: middle;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title = "Fashion Recommendation Chatbot"
st.write(
    "I'm here to help you find the perfect outfits! Ask me for recommendations to get started."
)

st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

chat_container = st.container()

with chat_container: 
    for msg in st.session_state.messages:
        if msg["role"] == "user": 
            st.markdown(
                f"""
                <div class="user-message">
                    <img src="https://img.icons8.com/fluency/48/user-male-circle.png" class="avatar"> 
                    {msg["content"]}
                </div>
            """,
                unsafe_allow_html=True,
            )

        else: 
            st.markdown(
                f"""
                <div class="bot-message">
                    <img src="https://img.icons8.com/?size=100&id=QIRhukOe1BpC&format=png&color=000000" class="avatar"> 
                    {msg["content"]}
                </div>
            """,
                unsafe_allow_html=True,
            )
        st.write("")


st.markdown("</div>", unsafe_allow_html=True)

query = st.text_input("Type your query here...", key="query")

col1, col2, col3 = st.columns([1, 3, 1])
with col2: 
    send_button = st.button("Get recommendation")

if send_button:
    if query:

        st.session_state.messages.append({"role":"user", "content":query})

        with chat_container: 
            st.markdown(
                f"""
                <div class="user-message">
                    <img src="https://img.icons8.com/fluency/48/user-male-circle.png" class="avatar"> 
                    {query}
                </div>
            """,
                unsafe_allow_html=True,
            )
            st.write("")

        with chat_container:
            st.markdown(
                '<div class="bot-message"><b>🤖 Thinking...</b></div>',
                unsafe_allow_html=True,
            )
            st.write("")

        try: 
            
            response = requests.post(API_URL + "/recommend", json={"question": query})
            
            if response.status_code == 200:
                answer = response.json().get("answer", "No recommendations found")
            else: 
                answer = "error: recommendation failed"

        except Exception as e:
            answer = f"Error: {str(e)}"

        with chat_container:
            st.markdown(
                f"""
                <div class="bot-message">
                    <img src="https://img.icons8.com/?size=100&id=XNotH4e8lEuO&format=png&color=000000" class="avatar"> 
                    {answer}
                </div>
            """,
                unsafe_allow_html=True,
            )
            st.write("")

        st.session_state.messages.append({"role":"assistant", "content":answer})


        
