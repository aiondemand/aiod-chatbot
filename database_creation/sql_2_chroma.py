import sqlite3
import json
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import chromadb
from chromadb.utils import embedding_functions


from dotenv import load_dotenv
import os
import platform

load_dotenv()  # This line brings all environment variables from .env into os.environ
api_key = os.environ['API_KEY']


def extract_table_data(db_path, table_name):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Prepare SQL query to select all content from the specified table
    query = f"SELECT * FROM {table_name}"

    # Execute the query
    cursor.execute(query)

    # Fetch all rows from the query result
    data = cursor.fetchall()

    # Close the cursor and the connection
    cursor.close()
    conn.close()

    # Return the list of data
    return data


def write_to_chroma(sql_db_name, sql_table_name, chroma_db_name):
    data_list = extract_table_data(sql_db_name, sql_table_name)
    all_documents = []

    for element in data_list:
        document_content = element[1] + "\n" + element[0] + "\n" + element[3] + "\n" + element[2]
        doc = Document(page_content=document_content, metadata={"source": element[1], "type": sql_table_name})
        all_documents.append(doc)

    # write to db
    Chroma.from_documents(all_documents, OpenAIEmbeddings(openai_api_key=api_key), persist_directory=chroma_db_name)


def write_to_chroma_cli(sql_db_name, sql_table_name):
    data_list = extract_table_data(sql_db_name, sql_table_name)
    all_documents = []

    content_list = []
    metadata_list = []
    id_list = []
    for index, element in enumerate(data_list):
        document_content = element[1] + "\n" + element[0] + "\n" + element[3] + "\n" + element[2]  # you might want to adapt this depending on your database structure
        content_list.append(document_content)
        metadata_list.append({"source": element[1], "type": sql_table_name})  # it is important that the content of the url column is saved as source in the metadata
        id_list.append("id_"+sql_db_name+str(index))
        # doc = Document(page_content=document_content, metadata={"source": element[1], "type": sql_table_name})
        # all_documents.append(doc)

    # write to db
    # uses openAI embeddings. If different embedding model is fancied,
    # change ef in all files where an embedding function is provided
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )

    chroma_vectorstore = chromadb.HttpClient(host='localhost', port=8000)
    my_library_resources = chroma_vectorstore.get_or_create_collection(name="my_library_resources", embedding_function=ef)

    n = 1000
    print(content_list[0])
    if len(content_list) > 1000:
        for i in range(0, len(content_list), n):
            my_library_resources.add(documents=content_list[i:i + n], metadatas=metadata_list[i:i + n], ids=id_list[i:i + n])

    else:
        print("")
        # my_library_resources.add(documents=content_list, metadatas=metadata_list, ids=id_list)


table_names = ['datasets', 'publications', 'educational_resources', 'experiments', 'ml_models']
# table_names = ['datasets']

if platform.system() == "Windows":
    sql_path = "./PycharmProjects/AI4Europe_new_UX/resources_2024-07-04.db"
else:
    sql_path = "resources_2024-07-04.db"
# Process each table
for table in table_names:
    write_to_chroma_cli(sql_path, table)


