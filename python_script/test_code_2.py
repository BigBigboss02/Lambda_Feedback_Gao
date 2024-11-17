import psycopg2
import pandas as pd

host = "aws-0-eu-west-2.pooler.supabase.com"  # Replace with your actual host
dbname = "postgres"  # Database name
user = "lf_zg819.ysvrjbquqkmctzytvwzd"  # Replace with your actual username
password = '="3!67v*SY1j'  # Replace with your actual password
port = "6543"  # Default PostgreSQL port
query = '''
SELECT * FROM public."Question"
ORDER BY "createdAt" DESC
LIMIT 10;
'''
conn = None

try:
    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        host=host,
        dbname=dbname,
        user=user,
        password=password,
        port=port
    )
    print("Connected to the database successfully.")

    # Query to get data from the Question table
    query = query

    # Execute the query and fetch data into a Pandas DataFrame
    df = pd.read_sql(query, conn)

    # Save as CSV
    df.to_csv(r"C:\Users\Malub.000\OneDrive - Imperial College London\Documents\OneDrive - Imperial College London\General - FYP 24_25 - Peter Johnson - ME\obtained_data\questions_data.csv", index=False)
    print("Data saved as CSV file.")

    # Save as JSON
    df.to_json(r"C:\Users\Malub.000\OneDrive - Imperial College London\Documents\OneDrive - Imperial College London\General - FYP 24_25 - Peter Johnson - ME\obtained_data\questions_data.json", orient="records", date_format="iso")
    print("Data saved as JSON file.")

except Exception as e:
    print("Failed to connect to the database or retrieve data.")
    print(e)

finally:
    # Close the database connection
    if conn is not None:
        conn.close()
        print("Database connection closed.")

