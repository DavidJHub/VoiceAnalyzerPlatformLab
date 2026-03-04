import re
from datetime import datetime, timedelta

import pandas as pd

import database.dbConfig as dbcfg
from utils.VapFunctions import get_latest_directories, clean_column_blank_regex
from utils.campaignMetrics import count_s3_audio_files

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


def get_file_count_in_folder(s3_url):
    """
    Given an S3 URL (e.g., s3://my-bucket/some/prefix/),
    return the total number of objects found under that prefix.
    """
    s3_client = dbcfg.generate_s3_client()

    if not s3_url or not isinstance(s3_url, str):
        return 0

    # Parse URL to bucket and prefix
    bucket, prefix = parse_s3_url(s3_url)

    # Paginator for large result sets
    paginator = s3_client.get_paginator('list_objects_v2')
    file_count = 0

    # Count all objects in the prefix
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            file_count += len(page['Contents'])

    return file_count


def get_file_count_in_folder_with_paths(s3_url, sub_paths_str, s3_client):
    """
    Given:
      - s3_url: e.g. s3://my-bucket/prefix/date_subfolder/
      - sub_paths_str: a string with comma-separated sub-paths, e.g. "foo, bar/baz"
      - s3_client: an authenticated boto3 S3 client

    This function sums the number of objects for each sub-path
    under the main 'date' prefix.
    """
    if not s3_url or not isinstance(s3_url, str):
        return 0
    if not sub_paths_str or not isinstance(sub_paths_str, str):
        return 0

    # -- 1) Parse the base S3 URL (the "most recent directory")
    bucket, prefix = parse_s3_url(s3_url)  # e.g. bucket="my-bucket", prefix="prefix/date_subfolder/"
    prefix = prefix.rstrip('/')  # remove trailing slash to avoid double slashes

    # -- 2) Split the sub_paths_str on commas, strip each
    sub_paths = [p.strip() for p in sub_paths_str.split(',') if p.strip()]

    # -- 3) For each sub-path, list objects that start with prefix + "/" + sub_path
    total_count = 0
    paginator = s3_client.get_paginator('list_objects_v2')

    for sp in sub_paths:
        # Ensure no leading/trailing slashes
        sp = sp.strip().strip('/')

        # Build the final prefix we want to list in S3, e.g. "prefix/date_subfolder/foo/"
        final_prefix = f"{prefix}/{sp}"
        if not final_prefix.endswith('/'):
            final_prefix += '/'

        # Go page by page to count all objects
        for page in paginator.paginate(Bucket=bucket, Prefix=final_prefix):
            if 'Contents' in page:
                total_count += len(page['Contents'])

    return total_count



def process_marketing_campaigns():
    """
    1. Query the marketing_campaigns table into a DataFrame.
    2. Clean the 's3' column.
    3. Create new columns:
        - latest_directories: all date-based subfolders
        - most_recent_directory: the newest date subfolder
        - file_count: sum of MP3 files (matching campaign_id) across all sub-paths in 'path'
    4. Return the modified DataFrame.
    """
    # --------------------------------------------------
    # A) Connect and read the entire table
    # --------------------------------------------------
    conn = dbcfg.conectar(dbcfg.HOST_DB_VAP,  
                                  dbcfg.PORT_DB_VAP,  
                                  dbcfg.DB_NAME_VAP,  
                                  dbcfg.USER_DB_VAP,  
                                  dbcfg.PASSWORD_DB_VAP)
    cursor = conn.cursor()

    query = "SELECT * FROM marketing_campaigns"
    cursor.execute(query)

    # Fetch rows and build a DataFrame
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]  # column names
    df = pd.DataFrame(rows, columns=columns)

    # Close cursor and connection
    cursor.close()
    conn.close()

    # --------------------------------------------------
    # B) Clean up any whitespace in the 's3' column
    # --------------------------------------------------
    if 's3' in df.columns:
        df = clean_column_blank_regex(df, 's3')

    # --------------------------------------------------
    # C) Create the S3 client
    # --------------------------------------------------
    s3_client = dbcfg.generate_s3_client()

    # --------------------------------------------------
    # D) For each row, parse S3 and get the latest directories
    # --------------------------------------------------
    def fetch_latest_dirs(s3_url):
        """
        Given an S3 URL, parse and get the latest directories using get_latest_directories().
        """
        if not s3_url or not isinstance(s3_url, str):
            return []
        bucket, prefix = parse_s3_url(s3_url)
        return get_latest_directories(bucket, prefix, s3_client)

    if 's3' in df.columns:
        df["latest_directories"] = df["s3"].apply(fetch_latest_dirs)
        df["most_recent_directory"] = df["latest_directories"].apply(
            lambda dirs: dirs[0] if dirs else None
        )

    # --------------------------------------------------
    # E) Count MP3 files for each row
    #
    #    We'll assume there is a column 'campaign_id' used to match in the S3 keys.
    #    We'll also assume there is a 'path' column that may contain multiple comma-separated sub-paths.
    # --------------------------------------------------
    def fetch_audio_count(row):
        """
        For a single row:
          - Take 'most_recent_directory' as the base prefix (date-based).
          - Split 'path' (comma-separated) into individual sub-paths.
          - For each sub-path, call count_s3_audio_files and sum the results.
        """
        # Ensure we have a valid directory and path
        base_s3_url = row.get("most_recent_directory")
        sub_paths_str = row.get("path", "")
        campaign_id = str(row.get("campaign_id", ""))  # or adjust if your ID column is named differently

        if not base_s3_url or not isinstance(base_s3_url, str):
            return 0

        # Parse the base URL
        bucket, base_prefix = parse_s3_url(base_s3_url)
        base_prefix = base_prefix.rstrip('/')  # remove trailing slash
        #print(base_prefix)

        # Split the sub-path string by comma
        sub_paths = [p.strip() for p in sub_paths_str.split(',') if p.strip()]

        total_audios = 0
        for sp in sub_paths:
            #print(sp)
            # Build the final prefix (e.g. "prefix/20240101/subpath/")
            # Remove leading/trailing slashes from sp

            # Count mp3 files that contain campaign_id in the Key
            total_audios += count_s3_audio_files(bucket, sp, base_prefix, s3_client)
            #print(total_mp3)

        return total_audios

    # Only proceed if we have columns for directory, path, and campaign_id
    if all(col in df.columns for col in ["most_recent_directory", "path", "campaign_id"]):
        df["file_count"] = df.apply(fetch_audio_count, axis=1)
    else:
        df["file_count"] = 0

    return df


def extract_subfolder_date_from_path(s3_url: str):
    """
    Given an S3 URL (e.g. s3://bucket/prefix/2023-10-12/ or s3://bucket/prefix/20231012/),
    extract the date from the last subfolder and return it as a date object.
    Returns None if no valid date is found.
    """
    if not s3_url or not isinstance(s3_url, str):
        return None

    # Remove any trailing slash
    s3_url = s3_url.rstrip('/')

    # The last segment should be the date subfolder
    parts = s3_url.split('/')
    last_part = parts[-1]  # e.g. 2023-10-12 or 20231012

    # Try matching yyyy-mm-dd
    if date_pattern_1.match(last_part):
        return datetime.strptime(last_part, '%Y-%m-%d').date()
    # Try matching yyyymmdd
    elif date_pattern_2.match(last_part):
        return datetime.strptime(last_part, '%Y%m%d').date()

    return None


def filter_subfolders_within_one_day(df: pd.DataFrame, directory_col: str = "most_recent_directory") -> pd.DataFrame:
    """
    Filters rows in `df` to keep only those whose subfolder date in `directory_col`
    is at most one day older than today's date.

    Steps:
      1. Extract date from the subfolder (e.g. 2023-10-12 or 20231012).
      2. Compare with (today - 1 day).
      3. Keep the row if subfolder_date >= (today - 1 day).
    """
    # Current date (no time) minus 1 day
    one_day_ago = datetime.now().date() - timedelta(days=2)

    # Extract the date from the most_recent_directory path
    df["subfolder_date"] = df[directory_col].apply(extract_subfolder_date_from_path)

    # Filter: keep only rows whose subfolder_date is within 1 day of "today"
    filtered_df = df[df["subfolder_date"] >= one_day_ago]

    # Optional: remove the helper column if you don't need it afterwards
    # filtered_df = filtered_df.drop(columns=["subfolder_date"])

    return filtered_df

def process_and_filter_marketing_campaigns():
    """
    1. Pull the marketing_campaigns data into a DataFrame.
    2. Generate columns: latest_directories, most_recent_directory, file_count.
    3. Filter rows whose most_recent_directory date is at most 1 day old.
    4. Return the filtered DataFrame.
    """
    df = process_marketing_campaigns()  # hypothetical function from previous steps

    df_filtered = filter_subfolders_within_one_day(df, directory_col="most_recent_directory")

    return df_filtered


def distribute_campaigns_to_machines(df, num_machines=4):
    """
    Distribute the rows in `df` across `num_machines` such that
    the sum of file_count is approximately balanced.

    Returns a list of DataFrames, one for each machine.
    """

    # Make a copy so we don't modify the original df
    df_sorted = df.copy()

    # 1) Sort descending by file_count
    df_sorted.sort_values(by="file_count", ascending=False, inplace=True)

    # 2) Initialize bins (machine loads and machine data)
    machine_loads = [0] * num_machines
    machine_rows = [[] for _ in range(num_machines)]

    # 3) Greedily assign each row
    for _, row in df_sorted.iterrows():
        # Find the machine with the smallest load
        min_index = machine_loads.index(min(machine_loads))
        # Assign the row to that machine
        machine_rows[min_index].append(row)
        # Update the load
        machine_loads[min_index] += row["file_count"]

    # 4) Convert each machine_rows list to a DataFrame
    dfs_for_machines = []
    for i in range(num_machines):
        machine_df = pd.DataFrame(machine_rows[i], columns=df.columns)
        dfs_for_machines.append(machine_df)

    return dfs_for_machines


def create_params_file(
        df,
        path_col='most_recent_directory',
        sort_col='file_count',
        order='asc',
        filename='params.txt'
):
    """
    Creates a text file (params.txt) that contains lines in the format:
      <path_value> 0
    If the path column has multiple comma-separated paths, only the first is used.

    sort_col: The column by which to sort the rows before writing (default 'file_count')
    order: 'asc' for ascending or 'desc' for descending (default 'asc')
    filename: The name of the output text file.
    """

    # Determine ascending boolean based on order argument
    # 'asc' => ascending=True, 'desc' => ascending=False
    if order.lower() == 'asc':
        ascending = True
    elif order.lower() == 'desc':
        ascending = False
    else:
        raise ValueError("order parameter must be either 'asc' or 'desc'.")

    # Sort the DataFrame
    df_sorted = df.sort_values(by=sort_col, ascending=ascending).copy()

    with open(filename, 'w') as txt_file:
        for _, row in df_sorted.iterrows():
            # Get the path from the DataFrame row
            path_value = row[path_col] if row[path_col] else ""

            # Convert to string and if multiple paths, take the first
            path_value = str(path_value).split(',')[0].strip()

            # Write the line: "<path_value> 0"
            txt_file.write(f"{path_value} 0\n")


if __name__ == "__main__":
    campaign_dataset=process_marketing_campaigns()
    date_pattern_1 = re.compile(r'^\d{4}-\d{2}-\d{2}$')  # Format yyyy-mm-dd
    date_pattern_2 = re.compile(r'^\d{8}$')  # Format yyyymmdd
    final_df = process_and_filter_marketing_campaigns()
    machine_dataframes = distribute_campaigns_to_machines(final_df, num_machines=4)

    # For each machine, create a params file sorted by file_count
    for i, df_machine in enumerate(machine_dataframes, start=1):
        filename = f"params_machine{i}.txt"
        create_params_file(
            df_machine,
            path_col='path',
            order='asc',
            sort_col='file_count',
            filename=filename
        )

        total_files = df_machine["file_count"].sum()
        print(f"Machine {i} => {len(df_machine)} rows, total file_count={total_files}")

