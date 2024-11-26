"""

Connect to Supabase PostgreSQL anonymous DB
Requires authorization from .env file
Extracts all part access events for all students in a given Module
Then isolates the "First" access event
The resulting dataframe provides the "createdAt" instance that corresponds to the time at which each student 
first accesses each of the set in the module

"""

from environs import Env
import pandas as pd 
import numpy as np 
import psycopg2
import getpass


# Create an instance of Env
env = Env()
env.read_env("lambdaFeedback\Sophia\.env")

# Connect to the database by getting connection parameters from hidden .env file 
def connect():
    """ Import DB username and password from .env file"""
    # access_string = '="3!67v*SY1j'

    DBusername = 'lf_zg819.ysvrjbquqkmctzytvwzd'
    DBpassword = '="3!67v*SY1j'
    print('Logging in as '+DBusername)


    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(
            host=env.str('ALPHA_HOST'),
            port=env.int('PORT', default=5432),
            database=env.str('DATABASE'),
            user=DBusername,
            password=DBpassword
        )

        return conn
    except Exception as e:
        print(f'Error connecting: {e}')
        return None

def queries(conn, module_name, module_instance_code):

    try:
        # create a cursor
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM public."QuestionAlgorithmStep"
            ORDER BY "variable" DESC
            LIMIT 10;

            """
            .format(moduleName = 'moduleName', moduleInstance='moduleInstanceCode'), (module_name, module_instance_code)
                    )

        # Dataframe that gives all the access part events for student users in the mechanics fluid module 
        df = pd.DataFrame(np.asarray(cur.fetchall()))
        df.columns=["Created at", "User Id", "Module Name", "Set Name", "Set Number", "Question Number", "Part Number"]
        df['createdAt'] = pd.to_datetime(df['Created at'])
        pd.set_option('display.max_columns', None)

        earliest_access = df.groupby(["User Id", "Set Number"])["createdAt"].idxmin()
        earliest_access_df = df.loc[earliest_access]
        return earliest_access_df

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)