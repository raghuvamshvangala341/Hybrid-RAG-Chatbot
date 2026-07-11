import streamlit as st


def initialize_chat():

    if "messages" not in st.session_state:

        st.session_state.messages = []


def display_chat():

    for message in st.session_state.messages:

        with st.chat_message(

            message["role"]

        ):

            st.markdown(

                message["content"]

            )


def add_user_message(content):

    st.session_state.messages.append(

        {

            "role": "user",

            "content": content

        }

    )


def add_assistant_message(content):

    st.session_state.messages.append(

        {

            "role": "assistant",

            "content": content

        }

    )