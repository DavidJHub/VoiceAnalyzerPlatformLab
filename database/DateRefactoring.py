import re
import boto3

from database.dbConfig import generate_s3_client

def rename_date_directories(
    bucket_name: str,
    prefix: str
) -> None:
    """
    Renames sub-'directories' (prefixes) in an S3 bucket from dd-mm-yyyy format to yyyy-mm-dd format.
    If a subdirectory is already yyyy-mm-dd, it is skipped.
    Any prefix that doesn't match either format is also skipped.

    :param bucket_name: The name of the S3 bucket.
    :param prefix: The S3 key prefix where the date-named directories are located (e.g., 'my-data/').
                   Make sure it includes any leading path you need, and typically ends with a slash.
    """

    # Compile regex patterns for dd-mm-yyyy/ and yyyy-mm-dd/ (with trailing slash)
    pattern_ddmmyyyy = re.compile(r'^(\d{2})-(\d{2})-(\d{4})/$')  # e.g., 01-02-2023/
    pattern_yyyymmdd = re.compile(r'^(\d{4})-(\d{2})-(\d{2})/$')  # e.g., 2023-02-01/

    s3_client = generate_s3_client()

    # List "subdirectories" (i.e., S3 common prefixes) immediately under the given prefix
    # Delimiter='/' instructs S3 to group objects by everything up to the first slash after prefix
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=prefix,
        Delimiter='/'
    )

    # Check if the response has 'CommonPrefixes' (these are the "directories")
    if 'CommonPrefixes' not in response:
        print("No subdirectories found under this prefix.")
        return

    for cp in response['CommonPrefixes']:
        old_subdir = cp['Prefix']         # e.g. 'my-data/01-02-2021/'
        subdir_name = old_subdir[len(prefix):]  # e.g. '01-02-2021/'

        # 1. Skip if already in yyyy-mm-dd format
        if pattern_yyyymmdd.match(subdir_name):
            print(f"Skipping already-correct format: {old_subdir}")
            continue

        # 2. If dd-mm-yyyy, rename to yyyy-mm-dd
        m = pattern_ddmmyyyy.match(subdir_name)
        if m:
            day, month, year = m.groups()  # e.g. ('01', '02', '2021')
            new_subdir_name = f"{year}-{month}-{day}/"  # e.g. '2021-02-01/'
            new_prefix = prefix + new_subdir_name

            print(f"Renaming {old_subdir} -> {new_prefix}")

            # Copy all objects from the old prefix to the new prefix, then delete the old objects
            paginator = s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket_name, Prefix=old_subdir):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        old_key = obj['Key']
                        # Construct the new key by replacing old_subdir with new_prefix at the front
                        new_key = new_prefix + old_key[len(old_subdir):]

                        # 1) Copy to new key
                        s3_client.copy_object(
                            CopySource={'Bucket': bucket_name, 'Key': old_key},
                            Bucket=bucket_name,
                            Key=new_key
                        )
                        # 2) Delete old key
                        s3_client.delete_object(Bucket=bucket_name, Key=old_key)

            print(f"Finished renaming {old_subdir} to {new_subdir_name}")
        else:
            # 3. If it doesn’t match either pattern, just skip
            print(f"Skipping unrecognized prefix: {old_subdir}")

if __name__ == "__main__":
    # Example usage
    bucket = "s3iahub.igs"
    prefix = "Guatemala/Bantrab/"  # must end with a slash if you're treating it like a folder

    rename_date_directories(bucket, prefix)
