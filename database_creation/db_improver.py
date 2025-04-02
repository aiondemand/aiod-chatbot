import sqlite3
import requests
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from dotenv import load_dotenv
import os

load_dotenv()  # This line brings all environment variables from .env into os.environ
openai.api_key = os.environ['API_KEY']


def add_enhancement_flag_column(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN enhanced TEXT")
    conn.commit()
    conn.close()


def fetch_web_data(url):
    a = str(url).split(".")
    if a[-1] == "pdf" or a[-1] == "PDF":
        try:
            pdf_loader = PyPDFLoader(url)
            content = pdf_loader.load()

            return content[0].page_content, 'Success'
        except Exception as e:
            return None, f'Failed: {str(e)}'
    else:
        try:
            headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0"}
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            html_content = response.text

            return html_content, 'Success'
        except Exception as e:
            return None, f'Failed: {str(e)}'


def generate_description_and_keywords(content, type_to_generate):
    if type_to_generate == 'description':
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    'role': 'user',
                    'content': f"Generate a brief description of two sentences or less from the following content: {content[:4000]}"}],
                max_tokens=200
            )
            text = response.choices[0].message.content
            return text, 'Success'

        except Exception as e:
            return 'No description found', f'Failed: {str(e)}'

    if type_to_generate == 'keywords':
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    'role': 'user',
                    'content': f"Generate a list of the most important keywords (max 5) from the following content: {content[:4000]}"}],
                max_tokens=200
            )
            text = response.choices[0].message.content
            return text, 'Success'

        except Exception as e:
            return 'No keywords found', f'Failed: {str(e)}'

    else:
        return '', '[]'


def update_database(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"SELECT rowid, * FROM {table_name} WHERE description IS '' OR description IS NULL OR description IS 'None' OR keywords IS '[]' OR keywords IS NULL")
    row_list = cursor.fetchall()
    for row in row_list:
        html_content, status = fetch_web_data(row[2])  # row[2] == link -> you might have to adapt this depending on how you saved your SQL db

        if status == 'Success':
            print("success")
            if row[3] == '' or row[3] == 'None':  # no description
                description, gen_status = generate_description_and_keywords(html_content, "description")
                status = gen_status
                cursor.execute(f"UPDATE {table_name} SET description = ?, enhanced = ? WHERE rowid = ?",
                               (description, status, row[0]))
                conn.commit()

            if row[4] == '[]':  # no keywords
                keywords, gen_status = generate_description_and_keywords(html_content, "keywords")
                status = gen_status
                cursor.execute(f"UPDATE {table_name} SET keywords = ?, enhanced = ? WHERE rowid = ?",
                               (keywords, status, row[0]))
                conn.commit()
    conn.close()


def rebuild_urls(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"SELECT rowid, * FROM {table_name} WHERE link like '%openml.org%'")
    row_list = cursor.fetchall()
    print(row_list[0])
    for row in row_list:
        row_link_list = row[2].split("/")
        new_link = f"https://www.openml.org/search?type={row_link_list[-2]}&id={row_link_list[-1]}"
        print(new_link)
        break
        # cursor.execute(f"UPDATE {table_name} SET link = ? WHERE rowid = ?", (new_link, row[0]))
        # conn.commit()


# Configuration
db_path = './PycharmProjects/AI4Europe_new_UX/resources_2024-07-04.db'
table_names = ['publications', 'educational_resources', 'experiments', 'ml_models', 'datasets']

# rebuild_urls(db_path, table_names)
# Process each table
for table in table_names:
    add_enhancement_flag_column(db_path, table)
    update_database(db_path, table)

"""conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute(f"SELECT rowid, * FROM ml_models WHERE description IS 'None'")
print(cursor.fetchall())"""
