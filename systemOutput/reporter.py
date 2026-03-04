import pandas as pd

from database.SQLDataManager import conectar
from database.dbConfig import generate_s3_client
from utils.VapFunctions import get_latest_directories, clean_column_blank_regex
import boto3

def parse_s3_url(s3_url: str):
    """
    Given an S3 URL of the form s3://bucket/prefix/...,
    return (bucket, prefix).
    """
    # Remove the scheme
    s3_url = s3_url.replace("s3://", "")

    # Split into bucket and the remaining path
    parts = s3_url.split("/", 1)
    bucket = parts[0]
    prefix = ""
    if len(parts) > 1:
        prefix = parts[1].rstrip("/") + "/"  # ensure trailing slash

    return bucket, prefix



def process_marketing_campaigns():
    """
    1. Query the marketing_campaigns table into a DataFrame.
    2. Create new columns with the latest directories for each S3 path.
    3. Return or save the modified DataFrame.
    """
    # --------------------------------------------------
    # 4a) Connect and read the entire table
    # --------------------------------------------------
    conn = conectar()
    cursor = conn.cursor()

    query = "SELECT * FROM marketing_campaigns"
    cursor.execute(query)

    # Fetch the rows and build a DataFrame
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]  # column names
    df = pd.DataFrame(rows, columns=columns)

    # Close the cursor and connection
    cursor.close()
    conn.close()

    # Clean up any whitespace in the 's3' column if needed
    if 's3' in df.columns:
        df = clean_column_blank_regex(df, 's3')

    # --------------------------------------------------
    # 4b) Create the S3 client
    # --------------------------------------------------
    s3_client = generate_s3_client()

    # --------------------------------------------------
    # 4c) For each row in the DF, parse the S3 path, call get_latest_directories
    # --------------------------------------------------
    def fetch_latest_dirs(s3_url):
        """
        Given an S3 URL, parse and get the latest directories using get_latest_directories().
        """
        if not s3_url or not isinstance(s3_url, str):
            return []

        bucket, prefix = parse_s3_url(s3_url)
        return get_latest_directories(bucket, prefix, s3_client)

    # Apply the function to each row in the DF
    if 's3' in df.columns:
        df["latest_directories"] = df["s3"].apply(fetch_latest_dirs)
        # Optionally, also store the first (most recent) directory:
        df["most_recent_directory"] = df["latest_directories"].apply(
            lambda dirs: dirs[0] if dirs else None
        )

    return df
