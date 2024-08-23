from chromadb.utils import embedding_functions
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from time import sleep
from uuid import uuid4
import streamlit as st
from dotenv import load_dotenv
import os
import platform
import chromadb

import sys
from datetime import datetime

from typing import Optional

if platform.system() == "Windows":
    pass
    # website_dir = "./aiod_db"
    # resources_dir = "./resources_db"
    # sys.stdout = open("conversation_logs/system_output_"+str(datetime.date(datetime.now()))+".txt", "a")
else:
    # website_dir = "./aiod_db_2024-07-24"
    # resources_dir = "./resources_db_2024-07-04"
    sys.stdout = open("conversation_logs/system_output_"+str(datetime.date(datetime.now()))+".txt", "a")

try:
    load_dotenv()  # This line brings all environment variables from .env into os.environ
    api_key = os.environ['API_KEY']

except FileNotFoundError:
    api_key = st.secrets['api_key']

# print(api_key)

# llm used
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
# website vectordb
# vectorstore = Chroma(persist_directory=website_dir, embedding_function=OpenAIEmbeddings(openai_api_key=api_key))
ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-3-small"
            )
chroma_vectorstore = chromadb.HttpClient(host='localhost', port=8000)
website_content = chroma_vectorstore.get_collection(name="aiod_website", embedding_function=ef)
# https://python.langchain.com/docs/modules/memory/agent_with_memory/
# modified memory to summarize previous messages ensure context length doesn't get out of control
#  possibilities: only store last n messages, summarize conversation history
#  https://python.langchain.com/docs/use_cases/chatbots/memory_management/

# https://python.langchain.com/docs/use_cases/tool_use/prompting/
# https://python.langchain.com/docs/use_cases/tool_use/multiple_tools/
# https://python.langchain.com/docs/modules/agents/
# https://python.langchain.com/docs/langgraph/
# https://python.langchain.com/docs/use_cases/sql/agents/
# https://python.langchain.com/docs/use_cases/sql/
# https://python.langchain.com/docs/modules/tools/
store = {}


def get_session_history_v2(session_id: str) -> str:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        return store[session_id]
    print("SESSION ID", session_id)
    # print("New HISTORY:", store[session_id])
    message_hist = store[session_id].messages[-1]
    print(message_hist)
    try:
        message_hist.content = message_hist.content.split("Final Answer: ")[1]
    except IndexError:
        message_hist.content = message_hist.content

    return store[session_id]


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def reset_history(session_id):
    history = ChatMessageHistory(session_id=session_id)
    history.clear()


class SearchInput(BaseModel):  # https://python.langchain.com/docs/modules/tools/custom_tools/
    query: str = Field(description="should be a search query")


def aiod_page_search(query: str) -> str:
    """Used to explain the AIoD website."""
    print("aiod_page_search:", str(datetime.now()), "query:", query)
    embedding_vector = OpenAIEmbeddings(openai_api_key=api_key).embed_query(query)
    # docs = website_content.similarity_search_by_vector(embedding_vector, k=5)
    docs = website_content.query(embedding_vector, n_results=5)
    # print(docs)
    # print(docs, type(docs))
    content_list = docs['documents'][0]
    metadata_list = docs['metadatas'][0]
    result = ""
    for index, doc in enumerate(content_list):
        result += "Result "+str(index) + ":\n" + doc + "\nlink:" + metadata_list[index]["source"] + "\n"
    print("tool output", result)
    return result


def database_similarity_search_v3(query: str) -> str:
    """
    Used to search for resources a user needs.
    :param query: user input
    :param session_id: user session id
    :return:
    """
    print(str(datetime.now()), "database_similarity_search_v3", "query:", query)
    allowed_datatypes = ["datasets", "publications", "educational_resources", "experiments", "ml_models"]
    dataset_dict = {
        "datasets": ["dataset", "datasets"],
        "publications": ["publication", "publications"],
        "educational_resources": ["educational_resources", "educational_resource", "educational resources", "educational resource"],
        "experiments": ["experiment", "experiments"],
        "ml_models": ["ml_model", "ml_models", "machine learning model", "machine learning models", "ml model", "ml models"]}
    data_type = ""
    for key in dataset_dict.keys():
        for dt in dataset_dict[key]:
            if dt in query.lower():
                data_type = key
                query = query.replace(dt, "")
                break
    embedding_vector = OpenAIEmbeddings(openai_api_key=api_key).embed_query(query)

    # print(docs)
    # print(docs, type(docs))
    if data_type not in allowed_datatypes:
        mylibrary_content = chroma_vectorstore.get_collection(name="my_library_resources", embedding_function=ef)
        docs = mylibrary_content.query(embedding_vector, n_results=5)
        # docs = mylibrary_content.similarity_search_by_vector(embedding_vector, k=5)
    else:
        print(str(datetime.now()), "data_type specific search")
        mylibrary_content = chroma_vectorstore.get_collection(name="my_library_resources", embedding_function=ef)
        docs = mylibrary_content.query(embedding_vector, n_results=5, where={"type": data_type})
        # docs = mylibrary_content.similarity_search_by_vector(embedding_vector, k=5, filter={"type": data_type})

    content_list = docs['documents'][0]
    metadata_list = docs['metadatas'][0]
    result = ""
    for index, doc in enumerate(content_list):
        result += "Result " + str(index) + ":\n" + doc + "\nlink:" + metadata_list[index]["source"] + "\n"
    # print("search result", result)
    print("tool output", result)
    return result  # , embedding_vector


"""ev_list = []
result_list = []
for i in range(50):
    result, ev = database_similarity_search_v3("datasets drones")
    ev_list.append(ev)
    result_list.append(result.replace("\n", ""))

with open("embedding_result.txt", "a") as f:
    for ev in ev_list:
        f.write(str(ev)+"\n")

with open("result.txt", "a") as f:
    for index, r in enumerate(result_list):
        f.write("Result "+str(index) + ": " + r + "\n")"""


def construct_doc_contains(query):
    keyword_list = query.split(" ")
    if len(keyword_list) > 1:
        result = []
        for element in keyword_list:
            result.append({"$contains": element})
        and_result = {"$and": result}
    else:
        and_result = {"$contains": keyword_list[0]}
    return and_result


def database_keyword_search(query: str) -> str:
    """
    Used to search for resources a user needs.
    :param query: user input
    :return:
    """
    print(str(datetime.now()), "database_keyword_search", "query:", query)
    allowed_datatypes = ["datasets", "publications", "educational_resources", "experiments", "ml_models"]
    dataset_dict = {
        "datasets": ["dataset ", "datasets "],
        "publications": ["publication ", "publications "],
        "educational_resources": ["educational_resources ", "educational_resource ", "educational resources ", "educational resource "],
        "experiments": ["experiment ", "experiments "],
        "ml_models": ["ml_model ", "ml_models ", "machine learning model ", "machine learning models ", "ml model ", "ml models "]}
    data_type = ""
    for key in dataset_dict.keys():
        for dt in dataset_dict[key]:
            if dt in query.lower():
                data_type = key
                query = query.replace(dt, "")
                break
    embedding_vector = OpenAIEmbeddings(openai_api_key=api_key).embed_query(query)

    contains = construct_doc_contains(query)
    # print(docs)
    # print(docs, type(docs))
    if data_type not in allowed_datatypes:
        mylibrary_content = chroma_vectorstore.get_collection(name="my_library_resources", embedding_function=ef)
        docs = mylibrary_content.query(embedding_vector, n_results=5, where_document=contains)
        # docs = mylibrary_content.similarity_search_by_vector(embedding_vector, k=5)
    else:
        mylibrary_content = chroma_vectorstore.get_collection(name="my_library_resources", embedding_function=ef)
        docs = mylibrary_content.query(embedding_vector, n_results=5, where={"type": data_type}, where_document=contains)
        # docs = mylibrary_content.similarity_search_by_vector(embedding_vector, k=5, filter={"type": data_type})
    print(docs['distances'])
    content_list = docs['documents'][0]
    metadata_list = docs['metadatas'][0]
    result = ""
    for index, doc in enumerate(content_list):
        result += "Result " + str(index) + ":\n" + doc + "\nlink:" + metadata_list[index]["source"] + "\n"
    # print("search result", result)
    print("tool output", result)
    return result  # , embedding_vector


# database_keyword_search("dataset tiger")
ps_desc = """Use the unmodified user input to get information about specific webpages on the AIoD website."""


class PageSimilaritySearch(BaseTool):
    name = "page_search"
    description = ps_desc
    # args_schema: Type[BaseModel] = SearchInput

    def _run(
        self,
        query: str,
    ) -> str: return aiod_page_search(query)

    async def _arun(
        self,
        query: str,
        data_type: Optional[str] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


rsd_desc = """Use this to search for resources that fit to the users requests. Available resource types are: publications, datasets, educational_resources, experiments, ml_models.
    """

rsd_desc_kw = """Use this to search for specific user requests in the resources. Searches for documents of the specified resource type where all words in the query are in the document. Available resource types are: publications, datasets, educational_resources, experiments, ml_models.
    """
rsd_args = """Example: user: I want to know something about drones
    Action: use resource_search
    Result:
    publications 1: Citizen Consultation | Drone-in-a-box by NAUST Robotics keywords: Robotics4EU, Citizen Consultation, Drone-in-a-box, NAUST Robotics, Publication
The HTML code contains metadata and links for a website related to Robotics4EU, focusing on citizen consultation and drone technology by NAUST Robotics. The webpage includes social media tags and structured data for SEO optimization.
link:https://www.robotics4eu.eu/publications/citizen-consultation-drone-in-a-box-by-naust-robotics/"""

rsd_desc_v2 = """Use this tool search for resources that fit to the users requests. Available resource types are: publications, datasets, educational_resources, experiments, ml_models. If you want to search for multiple resources at the same time, use keyword 'all'. 
    Make sure to give resource_search an argument structured like this: \{'input': 'the thing you want to search for', 'type': 'the type of the thing you want to search'\}
    Example: user: show me publications about drones
    Action: use resource_search with the input \{'input': 'drones', 'type': 'publications'\}
    Result:
    publications 1: Citizen Consultation | Drone-in-a-box by NAUST Robotics keywords: Robotics4EU, Citizen Consultation, Drone-in-a-box, NAUST Robotics, Publication
The HTML code contains metadata and links for a website related to Robotics4EU, focusing on citizen consultation and drone technology by NAUST Robotics. The webpage includes social media tags and structured data for SEO optimization.
link:https://www.robotics4eu.eu/publications/citizen-consultation-drone-in-a-box-by-naust-robotics/"""


class ResourceSimilaritySearchV3(BaseTool):
    name = "resource_similarity_search"
    description = rsd_desc
    # args_schema: Type[BaseModel] = SearchInput

    def _run(
        self,
        query: str,
    ) -> str: return database_similarity_search_v3(query)

    async def _arun(
        self,
        query: str,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class KeywordSearch(BaseTool):
    name = "resource_keyword_search"
    description = rsd_desc_kw
    # args_schema: Type[BaseModel] = SearchInput

    def _run(
        self,
        query: str,
    ) -> str: return database_keyword_search(query)

    async def _arun(
        self,
        query: str,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


tools = [PageSimilaritySearch(), ResourceSimilaritySearchV3(), KeywordSearch()]
# tools = [KeywordSearch()]
# tools = [PageSimilaritySearch(), ResourceSimilaritySearchV3()]
# tools = [ResourceSimilaritySearchV3()]


prefix = """You are an intelligent interactive assistant that manages the AI on Demand (AIoD) website. It is an amalgamation consisting of multiple parts. For example, you can up/download pre-trained AI models, find/upload scientific publications and access/provide a number of relevant datasets.

Your goal is to guide new users, that have not visited the website you manage, to the resources they want. 
To guide users you will first have to gather some information on the user.
1. You have to figure out what interests the user has that overlap with the content provided on the website
2. You have to figure out what the user wants to get from the website

You can ask the users questions as you see fit, but make sure to stay on topic and stop the questioning once you gathered sufficient amount of information to guide the user towards an initial goal or after asking 3 questions. If the user wants more help afterwards, make sure to provide it and keep asking questions where and when needed. 

To help the user navigate the website, you can provide links to pages you deem relevant. Always provide links to the ressources you talk about. You have access to the following tools:"""

prefix2 = """You are an intelligent interactive assistant that manages the AI on Demand (AIoD) website. 
The AIoD website consists of multiple parts. It has been created to facilitate collaboration, exchange and development of AI in Europe.
For example, users can up/download pre-trained AI models, find/upload scientific publications and access/provide a number of datasets.
It is your job to help the user navigate the website using the page_search or help the user find resources using the resource_search providing links to the websites/sources you are talking about.
Always provide links to the resources and websites you talk about. After your search, check carefully if the results contain the information you need to answer the question. If you cannot find the information you are searching for, reformulate the query by removing stop words or using synonyms.
Only if you have exhausted all other options, say: 'I found no results answering your question, can you reformulate it?'
You have access to the following tools:
"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

suffix_v2 = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix2,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

final_prompt = prompt

# print(final_prompt.template)


def add_outside_info_into_context(interests, publication_history):
    scholar_prefix = """You are an intelligent interactive assistant that manages the AI on Demand (AIoD) website. 
    The AIoD website consists of multiple parts. It has been created to facilitate collaboration, exchange and development of AI in Europe.
    For example, users can up/download pre-trained AI models, find/upload scientific publications and access/provide a number of datasets.
    It is your job to help the user navigate the website using the page_search or help the user find resources using the resource_search providing links to the websites/sources you are talking about.
    Always provide links to the resources and websites you talk about. If you cannot find information about something say: 'I have no information on this field, can you reformulate the question?'
    You are having a conversation with a person with the following interests: {interests}
    and publication history: {publication_history}
    You have access to the following tools:
    """.format(interests=interests, publication_history=publication_history)
    scholar_prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=scholar_prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )
    return scholar_prompt


def summarize_messages(session_identifier):
    history = ChatMessageHistory(session_id=uuid4())
    stored_messages = history.messages
    # print("STORED MESSAGES", stored_messages)
    if len(stored_messages) == 0:
        return False
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                "Distill the above chat messages into a single summary message. Include as many specific details as you can.",
            ),
        ]
    )
    summarization_chain = summarization_prompt | llm
    try:  # try to summarize. If that fails keep working without summary
        summary_message = summarization_chain.invoke({"chat_history": stored_messages})
    except:
        summary_message = stored_messages
    history.clear()
    history.add_message(summary_message)
    # print("HISTORY", history.messages)

    return True


def repeatedly_invoke(agent, input_query, config):
    try:
        response = agent.invoke({'input': input_query}, config=config)
    except:
        sleep(2)
        try:
            response = agent.invoke({'input': input_query}, config=config)
        except:
            sleep(4)
            try:
                response = agent.invoke({'input': input_query}, config=config)
            except Exception as e:
                response = {"output": "Something went wrong, please try again."}
                print(datetime.now(), e)
    return response


def clean_final_response(final_response_str: str, words: list) -> str:
    # if "Observation: " in final_response_str and "Final Answer: " in final_response_str:
    #    words.append("Observation: ")
    lines = final_response_str.splitlines()
    filtered_lines = [line for line in lines if not any(line.startswith(word) for word in words)]
    return "\n".join(filtered_lines)


def clean_final_response_v2(final_response_str: str, words: list) -> str:
    if "Final Answer:" in final_response_str:
        return final_response_str.split("Final Answer:")[-1]
    else:
        lines = final_response_str.splitlines()
        filtered_lines = [line for line in lines if not any(line.startswith(word) for word in words)]
    return "\n".join(filtered_lines)


def agent_response(input_query, session_identifier, personalization=None):
    # print("session id", session_identifier)
    agent = create_openai_tools_agent(llm, tools, final_prompt)
    """if personalization:
        agent = create_openai_tools_agent(llm, tools, personalization)
    else:
        agent = create_openai_tools_agent(llm, tools, prompt_without_history)"""

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=6,
        max_execution_time=60,  # seconds use 1 second for exception testing
        # early_stopping_method="generate"
        # generate is not implemented on langchain-side even though its documentation says otherwise
    )
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,  # lambda session_id: ChatMessageHistory(session_id=session_identifier),
        input_messages_key='input',
        history_messages_key='chat_history'
    )

    print("input query", input_query)
    # print("session id", session_identifier)
    # response = agent_with_chat_history.invoke({'input': input_query}, config={'configurable': {"session_id": session_identifier}})
    response = repeatedly_invoke(agent_with_chat_history, input_query, config={'configurable': {"session_id": session_identifier}})
    """try:
        response = agent_with_chat_history.invoke({'input': input_query},
                                                   config={'configurable': {"session_id": session_identifier}})
    except Exception as e:
        response = {"output": "Something went wrong, please try again."}
        print(datetime.now(), e)"""

    final_response = response['output']  # .split("Final Answer:")[-1]

    if final_response == 'Agent stopped due to max iterations.':
        final_response = response['intermediate_steps']

    if isinstance(final_response, list):

        try:
            new_input = final_response[-1][1]

            exception_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Answer the given question to the best of your ability using the provided context. Make sure to cite your sources.",
                    ),
                    MessagesPlaceholder(variable_name="history"),
                    ("assistant", "Context: {context}"),
                    ("human", "{input}"),
                ]
            )
            runnable = exception_prompt | llm
            with_message_history = RunnableWithMessageHistory(
                runnable,
                get_session_history,
                input_messages_key="input",
                history_messages_key="history",
            )
            final_response = with_message_history.invoke({"input": input_query, "context": new_input}, config={
                'configurable': {"session_id": session_identifier}}).content

        except:
            final_response = 'I encountered an error. Could you reformulate the question?' + str(final_response)

    final_r = clean_final_response_v2(final_response, ["Thought:", "Action:", "Action Input:", "Question:", "Observation:"])
    # clean_r = final_r.replace("Observation: ", "").replace("Final Answer: ", "")

    return final_r
