import sqlite3
import requests


def get_aiod_data(counter_api_url, data_api_url, limit, initial_offset):
    all_responses = []

    response = requests.get(counter_api_url)
    amount_of_entries = response.json()

    response = requests.get(data_api_url, params={'limit': limit, 'offset': initial_offset})
    if type(response.json()) != list:
        print(f"offset {initial_offset} is BROKEN", response.json())
        get_aiod_data(counter_api_url, data_api_url, limit, initial_offset+1)
    # print(response.json())
    # print(response.json()[0])
    all_responses += response.json()
    highest_id = response.json()[-1]['identifier']

    while highest_id < amount_of_entries:
        print(highest_id, amount_of_entries)
        response = requests.get(data_api_url, params={'limit': limit, 'offset': highest_id})

        try:
            highest_id = response.json()[-1]['identifier']
            all_responses += response.json()
        except:
            """print("new offset", highest_id)
            print(response.json())
            get_aiod_data(counter_api_url, data_api_url, limit, highest_id)"""
            break
    return all_responses


def store_responses_in_db(database_name, table_name, counter_api_url, data_api_url, limit, initial_offset):
    json_data = get_aiod_data(counter_api_url, data_api_url, limit, initial_offset)
    clean_json_list = []

    for entry in json_data:
        try:
            if "openml.org" in entry['same_as']:
                json_entry_list = entry['same_as'].split("/")
                entry['same_as'] = f"https://www.openml.org/search?type={json_entry_list[-2]}&id={json_entry_list[-1]}"

            try:
                if 'plain' not in entry['description'].keys():
                    clean_json = {
                        "name": entry["name"],
                        "link": entry['same_as'],
                        'description': "",  # description sometimes empty
                        'keywords': str(entry['keyword'])
                    }
                else:
                    clean_json = {
                        "name": entry["name"],
                        "link": entry['same_as'],
                        'description': entry['description']['plain'],  # description sometimes empty
                        'keywords': str(entry['keyword'])
                    }

                clean_json_list.append(clean_json)

            except KeyError:
                # only add correctly formatted entries to db
                print("could not store response due to formatting issues")
                pass
        except KeyError:
            # only add correctly formatted entries to db
            print("no 'same_as' column found")
            print(entry)
            pass


    # Step 1: Analyze the JSON objects to define the SQL table schema.
    # Assuming all JSON objects have the same keys and simple data types.
    keys = clean_json_list[0].keys()

    columns = ", ".join([f"{key} TEXT" for key in keys])

    # Step 2: Connect to SQLite database (this will create the file if it doesn't exist).
    conn = sqlite3.connect(database_name)
    c = conn.cursor()

    # Step 3: Create a table.
    # WARNING: This drops the table if it exists, you might want to handle this differently.
    if initial_offset == 0:
        c.execute(f"DROP TABLE IF EXISTS {table_name}")
        c.execute(f"CREATE TABLE {table_name} ({columns})")

    # Step 4: Insert JSON data into the table.
    for item in clean_json_list:
        placeholders = ", ".join(["?"] * len(keys))
        sql = f"INSERT INTO {table_name} ({', '.join(keys)}) VALUES ({placeholders})"
        c.execute(sql, tuple(item.values()))

    # Commit the changes and close the connection.
    conn.commit()
    conn.close()

    print("Data has been successfully inserted into the database.")


# successfully executed
# store_responses_in_db('resources.db', 'experiments', "https://api.aiod.eu/counts/experiments/v1", "https://api.aiod.eu/experiments/v1", 100)
# store_responses_in_db('resources.db', 'educational_resources', "https://api.aiod.eu/counts/educational_resources/v1", "https://api.aiod.eu/educational_resources/v1", 100)
#store_responses_in_db('resources.db', 'publications', "https://api.aiod.eu/counts/publications/v1", "https://api.aiod.eu/publications/v1", 100)
# store_responses_in_db('../resources.db', 'ml_models', "https://api.aiod.eu/counts/ml_models/v1", "https://api.aiod.eu/ml_models/v1", 1000, 0)  # 13801 entries

# doesn't work - needs additional code
db_path = './PycharmProjects/AI4Europe_new_UX/resources_2024-07-26.db'
store_responses_in_db(db_path, 'datasets', "https://api.aiod.eu/counts/datasets/v1", "https://api.aiod.eu/datasets/v1", 200, 0)  # over 400k entries # 14352, 15612
store_responses_in_db(db_path, 'experiments', "https://api.aiod.eu/counts/experiments/v1", "https://api.aiod.eu/experiments/v1", 100, 0)
store_responses_in_db(db_path, 'educational_resources', "https://api.aiod.eu/counts/educational_resources/v1", "https://api.aiod.eu/educational_resources/v1", 100, 0)
store_responses_in_db(db_path, 'publications', "https://api.aiod.eu/counts/publications/v1", "https://api.aiod.eu/publications/v1", 100, 0)
store_responses_in_db(db_path, 'ml_models', "https://api.aiod.eu/counts/ml_models/v1", "https://api.aiod.eu/ml_models/v1", 1000, 0)

# response = requests.get("https://api.aiod.eu/datasets/v1", params={'limit': 3, 'offset': 121})
# response = requests.get("https://api.aiod.eu/ml_models/v1", params={'offset': 20})
# print(response.json())
