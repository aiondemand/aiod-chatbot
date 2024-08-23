# Talk2AIoD

## Getting Started

To use this repository, follow the steps below to set up and run the project. This work uses OpenAI models. You will need 
to either provide an OpenAI API key in a .env file with the content API_KEY = your_api_key or adapt all the places we used
an OpenAI LLM or embedding model to the models you want to use. If you plan to do that, first make sure that the LLM you 
want to use can use tools and is supported by LangChain.

### 1. Create the Database

First, create the database using ChromaCLI. This is the vector database the model will look into.

Start ChromaCLI by running the following command:

```sh
chroma run --host localhost --port 8000 --path path-where-you-want-to-store-the-data
```
Look at https://docs.trychroma.com/guides for further guidance.
### 2. Fill the Database
#### Crawl a Website
If you have a website to crawl, use scraper.py to create a collection in the ChromaDB storing the information available 
on the website. Make sure to set your starting point and anchor correctly to avoid crawling pages that should not be 
part of the collection.

#### Get data through an API
If the website you want to talk with has a database in the background you can reach through an API, do the following 
steps to create a collection in the ChromaDB. 
- Use the request_2_sql.py file to create a local SQLite database storing the information you want from the API endpoint
  (we are using the requests library for this, but since API endpoints tend to differ significantly you might need to
write your own code for this. It doesn't really matter how you get to the SQLite database, only that you get there.)
- Now you can use the db_improver.py to have an LLM try to fill in empty fields in the tables of your db. Notice that 
this is only possible if one of your table columns is an url-column that points to a website that can be crawled. You 
might have to change the columns where the db_improver expects the url to be found.

The sql_2_chroma.py will create a collection in your ChromaDB from the SQLite database.

#### The intelligent agent backend
In this file, the intelligent agents functionality and behavior is defined. If you want to change the used prompts or 
the tools used by the agent, you will find them here.

#### main.py
The streamlit frontend of the intelligent agent.

#### running the agent
to run the agent make sure your ChromaDB is running. Then start streamlit using this command:
```sh
streamlit run main.py
```
