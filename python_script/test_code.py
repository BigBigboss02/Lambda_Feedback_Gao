import psycopg2
import pandas as pd

# Database connection details
host = "aws-0-eu-west-2.pooler.supabase.com"  # Replace with your actual host
dbname = "Postgres"  # Database name
user = "lf_zg819.ysvrjbquqkmctzytvwzd"  # Replace with your actual username
password = '="3!67v*SY1j'  # Replace with your actual password
port = "6543"  # Default PostgreSQL port

# Connect to the PostgreSQL database
try:
    conn = psycopg2.connect(
        host=host,
        dbname=dbname,
        user=user,
        password=password,
        port=port
    )
    print("Connected to the database successfully.")
except Exception as e:
    print("Failed to connect to the database.")
    print(e)
    exit()

# Create a cursor object
cursor = conn.cursor()

# Sample query to retrieve data from a specific table (e.g., Module)
#query = "SELECT * FROM public.Module;"
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

# try:
#     # Execute the query
#     cursor.execute(query)

#     # Fetch all results
#     records = cursor.fetchall()

#     # If you want to load data into a Pandas DataFrame
#     colnames = [desc[0] for desc in cursor.description]  # Get column names
#     df = pd.DataFrame(records, columns=colnames)
#     print("Data retrieved successfully.")
#     print(df.head())  # Display the first few rows of the DataFrame
# except Exception as e:
#     print("Failed to retrieve data.")
#     print(e)

# Close the cursor and connection
cursor.close()
conn.close()
print("Database connection closed.")
