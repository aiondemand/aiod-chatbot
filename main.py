import intelligent_agent_backend
import streamlit as st
import get_scholar_data
import pandas as pd
from datetime import datetime
# from threading import Thread
# from streamlit.runtime.scriptrunner import add_script_run_ctx
# from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
# import time
# from streamlit_autorefresh import st_autorefresh


st.set_page_config(page_title="Chatbot")
st.title("Talk2AIoD")
"""This Chatbot was built to guide you through the AIoD website and recommend you resources for your work."""

if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

if 'interests' not in st.session_state:
    st.session_state['interests'] = []

if 'publications' not in st.session_state:
    st.session_state['publications'] = []

if 'scholar_prompt' not in st.session_state:
    st.session_state['scholar_prompt'] = []

if "time_update" not in st.session_state:
    st.session_state['time_update'] = []


#def tt2():
#    count = st_autorefresh(interval=1000, limit=60, key="fizzbuzzcounter")
#    placeholder = st.empty()
#    while count < 30:
#        placeholder.text(str(count))
#        if count == 5:
#            placeholder.text("I am searching...")
#        if count == 10:
#            placeholder.text("I am searching...\nThis might take a while.")
#        if count == 20:
#            placeholder.text("I am searching...\nThis might take a while.\nI am still not happy with the results")


def generate_background(input_text):
    with st.spinner("Looking through Google Scholar"):
        keywords, response, clean_response = get_scholar_data.get_author_result(input_text)
        interests = "Interests: "
        for keyword in keywords:
            interests += keyword + ", "
        st.text(interests[:-2])
        columns = ['index', 'title', 'year', 'citations']
        df = pd.DataFrame(clean_response, columns=columns).set_index('index')
        st.table(df)
    st.session_state['interests'] = interests
    st.session_state['publications'] = clean_response
    if len(clean_response) > 5:
        selected_responses = clean_response[:5]
    else:
        selected_responses = clean_response
    st.session_state['scholar_prompt'] = intelligent_agent_backend.add_outside_info_into_context(interests, selected_responses)
    try:
        uid = st.query_params['id']
    except KeyError:
        uid = -1
    path = "conversation_logs/" + str(uid) + ".txt"
    with open(path, 'a', encoding='utf-8') as f:
        f.write(str(datetime.now()) + "\n")
        f.write("#Google Scholar profile:\n")
        f.write(input_text)
        f.write("\n")
        f.write("#Interests:\n")
        f.write(interests)
        f.write("#Publications(max. 5):\n")
        for response in selected_responses:
            f.write(str(response)+"\n")
        f.write("\n\n#--------------------------\n\n")


def generate_response(input_text):
    # st.info(st.session_state['scholar_prompt'])
    with st.spinner("Working on it. This might take a while. Don't worry I'm making progress..."):
        try:
            uid = st.query_params['id']
        except KeyError:
            uid = "-1"
            st.query_params['id'] = "-1"
        if st.session_state['scholar_prompt']:
            response = intelligent_agent_backend.agent_response(input_text, uid, st.session_state['scholar_prompt'])
        else:
            # entertainer = Thread(target=tt2, args=())
            # add_script_run_ctx(entertainer)
            # entertainer.start()
            response = intelligent_agent_backend.agent_response(input_text, uid)

        st.info(response)
    st.session_state['conversation'].append((input_text, response))
    # print("generate response session state", st.session_state['conversation'])
    path = "conversation_logs/" + str(uid) + ".txt"
    with open(path, 'a', encoding='utf-8') as f:
        f.write("#Answer ("+str(datetime.now())+"):\n")
        f.write(response)
        f.write("\n\n#--------------------------\n\n")


def reset_all():
    # print("reset all")
    try:
        uid = st.query_params['id']
    except KeyError:
        uid = "-1"
    intelligent_agent_backend.reset_history(uid)
    st.session_state['conversation'] = []
    st.session_state['interests'] = []
    st.session_state['publications'] = []
    st.session_state['scholar_prompt'] = []


with st.form("my_form"):
    text = st.text_area("Enter text:", placeholder="Tell me about AIoD.")
    st.write(":red[Talk2AIoD can make mistakes. Verify important information.]")
    submitted = st.form_submit_button("Submit")
    if submitted and text:
        try:
            user_id = st.query_params['id']
        except KeyError:
            user_id = "-1"
        path = "conversation_logs/" + str(user_id) + ".txt"
        with open(path, 'a', encoding='utf-8') as f:
            f.write("#Question ("+str(datetime.now())+"):\n")
            f.write(text)
            f.write("\n")
        generate_response(text)


# with st.container():
#    st.write("Settings")
#    st.button("Forget Conversation", on_click=reset_all)
#    with st.form("my_form2"):
#        text = st.text_input("Provide your Google Scholar information:", placeholder="Albert Einstein")
#        submitted = st.form_submit_button("Submit")
#        if submitted and text:
#            generate_background(text)


st.sidebar.write("Previous questions and answers")
with st.sidebar:
    for element in st.session_state['conversation']:
        if len(element[0]) > 0:
            with st.sidebar.popover(element[0]):
                st.info(element[1])
