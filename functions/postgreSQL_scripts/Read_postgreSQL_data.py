"""

Connect to Supabase PostgreSQL anonymous DB
Requires authorization from .env file
Extracts all part access events for all students in a given Module
Then isolates the "First" access event
The resulting dataframe provides the "createdAt" instance that corresponds to the time at which each student 
first accesses each of the set in the module

"""

import pandas as pd
import psycopg2
from datetime import datetime
from dotenv import load_dotenv
import os

# Specify the path to the renamed .env file
env_path = 'login_configs.env'
basepath = 'test_results/database_clips'
load_dotenv(dotenv_path=env_path)

# Configuration class to store database connection details
class Config:
    DB_USERNAME = os.getenv("DB_USERNAME")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_NAME = os.getenv("DB_NAME")
    DB_PORT = os.getenv("DB_PORT")
Config = Config()

# Connect to the database using values from the Config class
def connect():
    """ Connect to the PostgreSQL database server """
    DBusername = Config.DB_USERNAME
    DBpassword = Config.DB_PASSWORD
    host = Config.DB_HOST
    dbname = Config.DB_NAME
    port = Config.DB_PORT

    conn = None
    try:
        # Connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=dbname,
            user=DBusername,
            password=DBpassword
        )
        print("Connected successfully.")
        return conn
    except Exception as e:
        print(f'Error connecting: {e}')
        return None

# Define the query
query = '''
SELECT
  p."partContent",
  p."partAnswerContent",
  qv."masterContent"

FROM
  "ResponseArea" AS ra
JOIN
  "EvaluationFunction" AS ef ON ra."evaluationFunctionId" = ef.id
JOIN
  "Part" AS p ON ra."partId" = p.id
JOIN
  "QuestionVersion" AS qv ON p."questionVersionId" = qv.id
JOIN
  "Question" AS q ON qv."questionId" = q.id
WHERE
  ef."name" = 'shortTextAnswer'
  AND q."publishedVersionId" = qv.id;

'''
query2 ='''
SELECT
  "ResponseArea".*,
  "EvaluationFunction".*,
  "Part".*,
  "QuestionVersion".*,
  "QuestionAlgorithmFunction".*
FROM
  "ResponseArea",
  (
    SELECT * FROM "EvaluationFunction"
    WHERE "name" = 'shortTextAnswer'
  ) AS "EvaluationFunction",
  (
    SELECT * FROM "Part"
    WHERE "Part"."id" = "ResponseArea"."partId"
  ) AS "Part",
  (
    SELECT * FROM "QuestionVersion"
    WHERE "QuestionVersion"."id" = "Part"."questionVersionId"
  ) AS "QuestionVersion",
  (
    SELECT * FROM "Question"
    WHERE "Question"."publishedVersionId" = "QuestionVersion"."id"
  ) AS "Question",
  (
    SELECT * FROM "QuestionAlgorithmFunction"
    WHERE "QuestionAlgorithmFunction"."id" = "Part"."questionAlgorithmFunctionId"
  ) AS "QuestionAlgorithmFunction"
WHERE
  "ResponseArea"."evaluationFunctionId" = "EvaluationFunction".id
  AND "Part"."questionVersionId" = "QuestionVersion".id
  AND "QuestionVersion"."questionId" = "Question".id;
'''

# def queries(conn, module_name, module_instance_code):

#     try:
#         # create a cursor
#         cur = conn.cursor()
#         cur.execute(
#                 """
#             with "ModuleOfInterest" as (
#             select * from "PartInfo1" 
#             where "{moduleName}"=%s and "{moduleInstance}"=%s
#             ),
#             "StudentsOnly" as (
#             select "id" from "User" 
#             where "role"='STUDENT'
#             ),
#             "PartsAccessed" as (
#             select * from "PartAccessEvent" 
#             where "universalPartId" in (select "universalPartId" from "ModuleOfInterest")
#             and "userId" in (select "id" from "StudentsOnly")
#             ), 
#             "data" as (
#             select "PartsAccessed".*, "ModuleOfInterest".*
#             from "PartsAccessed" 
#             join "ModuleOfInterest" on "PartsAccessed"."universalPartId" = "ModuleOfInterest"."universalPartId"
#             )
#             select "createdAt", "userId", "moduleName", "setName", "setNumber", "questionNumber", "partPosition"  from "data" 
#             ;
#             """
#             .format(moduleName = 'moduleName', moduleInstance='moduleInstanceCode'), (module_name, module_instance_code)
#                     )

#         # Dataframe that gives all the access part events for student users in the mechanics fluid module 
#         df = pd.DataFrame(np.asarray(cur.fetchall()))
#         df.columns=["Created at", "User Id", "Module Name", "Set Name", "Set Number", "Question Number", "Part Number"]
#         df['createdAt'] = pd.to_datetime(df['Created at'])
#         pd.set_option('display.max_columns', None)

#         earliest_access = df.groupby(["User Id", "Set Number"])["createdAt"].idxmin()
#         earliest_access_df = df.loc[earliest_access]
#         return earliest_access_df

#     except (Exception, psycopg2.DatabaseError) as error:
#         print(error)

# Fetch data using the connection and save it to a DataFrame
def fetch_data():
    conn = connect()
    if conn is not None:
        try:
            # Execute query and fetch data
            df = pd.read_sql_query(query, conn)
            print("Data fetched successfully.")
            print(df)  # Print the DataFrame
            # Display only the relevant columns: 'documentationContent', 'partContent', and 'partAnswerContent'
            #filtered_data = df[['documentationContent', 'postResponseText', 'preResponseText']].copy()

            # Renaming columns to match user request: 'postResponseText' -> 'partContent', 'preResponseText' -> 'partAnswerContent'
            # filtered_data.rename(columns={
            #     'postResponseText': 'partContent',
            #     'preResponseText': 'partAnswerContent'
            # }, inplace=True)
            # Get the current system time for the filename
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(basepath, f"data_{current_time}.csv")
            # Save CSV with current system time
            df.to_csv(file_path, index=False)

            print("Data saved to 'data.csv'.")
        except Exception as e:
            print(f"Error fetching data: {e}")
        finally:
            conn.close()
            print("Connection closed.")
    else:
        print("Failed to connect to the database.")

# Run the fetch_data function to execute everything
fetch_data()
