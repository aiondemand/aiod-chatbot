import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from playwright.sync_api import sync_playwright, Playwright
import chromadb
from chromadb.utils import embedding_functions

from dotenv import load_dotenv
import os



load_dotenv()  # This line brings all environment variables from .env into os.environ
api_key = os.environ['API_KEY']


def crawl_website(link_to_website, anchor_list):
    """
    Crawls a website saving all unique links it can find in a list and returning it.
    :param link_to_website: start of the crawl: string
    :param anchor_list: a list of strings where at least one element in the list has to be in the url
    :return: list(str(url))
    """
    urls = [link_to_website]
    pages = []

    # until all pages have been visited
    while len(urls) != 0:
        # get the page to visit from the list
        current_url = urls.pop()

        # crawling logic

        response = requests.get(current_url)
        soup = BeautifulSoup(response.content, "html.parser")

        link_elements = soup.select("a[href]")
        for link_element in link_elements:
            url = link_element['href']
            # print(url)
            for anchor in anchor_list:
                if anchor in url and "http" in url:
                    already_visited = False
                    for page in pages:
                        if page == url:
                            already_visited = True
                            break

                    if not already_visited:
                        urls.append(url)
                        pages.append(url)

    return pages


def str_2_list(path_to_file):
    with open(path_to_file, 'r') as f:
        content = f.read()

    a = content.replace("[", "").replace("]", "").replace("'", "")
    b = a.split(",")

    return b


def better_loader(url):
    """
    Actually waits until the page is fully loaded
    :param url: str
    :return: langchain Document object
    """
    def run(playwright: Playwright):
        chromium = playwright.chromium  # or "firefox" or "webkit".
        browser = chromium.launch()
        page = browser.new_page()
        page.goto(url)
        # other actions...
        page.wait_for_load_state("networkidle")
        page.once("load", lambda: print("page loaded!"))
        html_content = page.content()
        browser.close()
        return html_content

    with sync_playwright() as playwright:
        returned_html_content = run(playwright)

    doc = Document(page_content=str(url) + "\n" + returned_html_content, metadata={"source": url})
    return doc


def merge_documents(doc_list):
    source = doc_list[0].metadata['source']
    content_list = str(source) + "\n"
    for doc in doc_list:
        content_list += doc.page_content

    doc = Document(page_content=content_list, metadata={"source": source})

    return [doc]


def infuse_with_url(doc_list):
    infused_doc_list = []
    for doc in doc_list:
        source = doc.metadata['source']
        content = doc.page_content
        infused_doc = Document(page_content=str(source) + "\n" + content, metadata={"source": source})
        infused_doc_list.append(infused_doc)

    return infused_doc_list


def write_to_chroma(list_of_urls, db_name):

    urls = []
    pdfs = []
    for url in list_of_urls:
        a = str(url).split(".")
        if a[-1] == "pdf" or a[-1] == "PDF":
            pdfs.append(url)
        else:
            urls.append(url)

    print(pdfs)

    bs_transformer = BeautifulSoupTransformer()
    # Load HTMLs
    # loader = AsyncChromiumLoader(urls)
    # html = loader.load()

    html = []
    for url in urls:
        loader = better_loader(url)
        html.append(loader)

    # extract information from HTMLs
    docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["span", "p", "li", "div", "a", "label"])

    all_pages = []
    for pdf_path in pdfs:
        pdf_loader = PyPDFLoader(pdf_path)
        pages = pdf_loader.load_and_split()
        all_pages += merge_documents(pages)
 
    # chunk documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(docs_transformed)
    docs_infused = infuse_with_url(documents)

    all_documents = docs_infused + all_pages
    # write to db
    Chroma.from_documents(all_documents, OpenAIEmbeddings(openai_api_key=api_key), persist_directory=db_name)


def write_to_chroma_cli(list_of_urls, collection_name):
    """
    Getting a list of urls, this function looks at the website and extracts all information it can from html and pdf.
    It then embeds this information and writes it as a collection into a chromaDB. Uses openAI embeddings.
    :param list_of_urls: list(str(url))
    :param collection_name: str: name of the collection
    :return:
    """
    urls = []
    pdfs = []
    for url in list_of_urls:
        a = str(url).split(".")
        if a[-1] == "pdf" or a[-1] == "PDF":
            pdfs.append(url)
        else:
            urls.append(url)

    print(pdfs)

    bs_transformer = BeautifulSoupTransformer()
    # Load HTMLs
    # loader = AsyncChromiumLoader(urls)
    # html = loader.load()

    html = []
    for url in urls:
        loader = better_loader(url)
        html.append(loader)

    # extract information from HTMLs
    docs_transformed = bs_transformer.transform_documents(html,
                                                          tags_to_extract=["span", "p", "li", "div", "a", "label"])

    all_pages = []
    for pdf_path in pdfs:
        pdf_loader = PyPDFLoader(pdf_path)
        pages = pdf_loader.load_and_split()
        all_pages += merge_documents(pages)
    # print("all pages", all_pages)
    # chunk documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(docs_transformed)
    docs_infused = infuse_with_url(documents)
    # print("docs infused", docs_infused)
    all_documents = docs_infused + all_pages
    # write to db
    # print("hi")
    metadata_list = []
    content_list = []
    id_list = []
    max_len = 12000
    for index, document in enumerate(all_documents):
        content = document.page_content[:max_len]
        content_list.append(content)
        metadata_list.append(document.metadata)
        id_list.append("id"+str(index))

    # uses openAI embeddings. If different embedding model is fancied,
    # change ef in all files where an embedding function is provided
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )

    chroma_vectorstore = chromadb.HttpClient(host='localhost', port=8000)
    # chroma_vectorstore.delete_collection(name="aiod_website")
    website_content = chroma_vectorstore.get_or_create_collection(name=collection_name, embedding_function=ef)  # , metadata={"hnsw:space": "cosine"}
    website_content.add(documents=content_list, metadatas=metadata_list, ids=id_list)
    # Chroma.from_documents(all_documents, OpenAIEmbeddings(openai_api_key=api_key), persist_directory=db_name)


result = crawl_website("https://aiod.eu/", ["aiod.eu"])
with open("results/crawl_result_v8.txt", "w") as file:
    file.write(str(result))
print(result)

write_to_chroma_cli(result, "aiod_website")
