#!/usr/bin/env python3
"""
Copy objects whose filenames appear in an Excel column from one S3 prefix to another.

Requirements
------------
pip install boto3 pandas openpyxl

AWS credentials
---------------
The script uses the default AWS credential chain (env vars, ~/.aws/credentials, or an attached IAM role).
"""


from __future__ import annotations
import logging
import sys
from pathlib import PurePosixPath, Path
from typing import Iterable

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from database.dbConfig import generate_s3_client, generate_s3_resource

# ────────────────────────────────────────────────────────────────────────────────
# CONFIG —----------------------------------------------------------------------
# Edit only what is necessary; everything else should “just work”.
# ────────────────────────────────────────────────────────────────────────────────
EXCEL_PATH: Path | str = "Davi.xlsx"        # your Excel file
SHEET_NAME: str | int = 0                      # sheet index or name
COLUMN_NAME: str = "Nombre de llamada"                  # column that holds the filenames

SRC_BUCKET: str = "s3iahub.igs"           # source bucket
SRC_PREFIX: str = "Colombia/Davivienda/"                 # source “subdirectory” (can be empty string)

DST_BUCKET: str = "s3iahub.igs"             # destination bucket (can be same as SRC_BUCKET)
DST_PREFIX: str = "Colombia/Davivienda/2999-09-09/"           # destination “subdirectory”

AWS_REGION: str | None = "us-east-1"                   # set to e.g. "us-east-1" or leave None for default
MAX_KEYS_PER_LIST = 1_000                      # pagination chunk when enumerating big prefixes
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# INPUT PROCESSING ────────────────────────────────────────────────────────────────────────────────

import re, unicodedata

def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # unicode
    text = text.strip()                         # espacios extremos
    text = re.sub(r"\s+", "", text)             # tab / \r / \n internos
    return text.lower()                         # case-fold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


s3= generate_s3_resource()
s3_client = generate_s3_client()

def read_filenames_from_excel(
    path: Path | str,
    sheet: int | str = 0,
    column: str = "filename",
) -> list[str]:
    """Return a list of filenames (as strings) from an Excel column,
    dropping blanks and NaNs."""
    df = pd.read_excel(path, sheet_name=sheet)
    df['parsing_name'] = df[column]+"-all.mp3"
    series=(        df['parsing_name']
          .dropna()
          .astype(str)
          .map(clean)
          .loc[lambda s: s != ""]
          .unique())
    return series[series != ""].tolist()


def list_objects_once(bucket: str, prefix: str) -> list[dict]:
    """List *all* objects under a prefix (multiple API calls if needed)."""
    paginator = s3_client.get_paginator("list_objects_v2")
    logging.info("Scanning s3://%s/%s ... this may take a while.", bucket, prefix)
    keys: list[dict] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, PaginationConfig={"PageSize": MAX_KEYS_PER_LIST}):
        keys.extend(page.get("Contents", []))
    logging.info("Found %d objects under prefix.", len(keys))
    return keys


def build_lookup(objs: Iterable[dict]) -> dict[str, str]:
    """Return a mapping {filename -> full S3 key} for quick lookup."""
    lut = {}
    for obj in objs:                                      # ← aquí va tu bucle
        key = obj["Key"]
        filename = clean(Path(key).name)                  # se limpia SOLO para comparar
        lut.setdefault(filename, key)                     # pero se guarda el key real
    return lut


def copy_objects(filenames: list[str], src_lut: dict[str, str]) -> None:
    missing: list[str] = []
    copied: int = 0

    for filename in filenames:
        src_key = src_lut.get(filename)
        if not src_key:
            missing.append(filename)
            continue

        dst_key = f"{DST_PREFIX.rstrip('/')}/{Path(src_key).name}"
        copy_source = {"Bucket": SRC_BUCKET, "Key": src_key}



    # Summary
    logging.info("Copied %d objects.", copied)
    if missing:
        logging.warning("Did not find %d filenames:\n%s", len(missing), "\n".join(missing).upper())


def main() -> None:
    filenames = read_filenames_from_excel(EXCEL_PATH, sheet=SHEET_NAME, column=COLUMN_NAME)
    if not filenames:
        logging.error("No filenames found in column '%s'. Exiting.", COLUMN_NAME)
        return

    objects = list_objects_once(SRC_BUCKET, SRC_PREFIX)

    # ------------------------------------------------------------------
    print("Excel rows read :", len(filenames))
    print("First 5 Excel   :", [repr(x) for x in filenames[:5]])

    print("First 5 S3 keys :", [o["Key"] for o in objects[:5]])

    # Build *temporary* easy-to-read sets for comparison
    excel_set = {f.lower().strip() for f in filenames}
    s3_set    = {Path(o["Key"]).name.lower() for o in objects}

    matches = excel_set & s3_set
    print("Exact filename matches found :", len(matches))
    if len(matches) < 5:
        print("Sample mismatches Excel→S3:")
        for x in list(excel_set - s3_set)[:5]:
            print("  ", repr(x))
    # ------------------------------------------------------------------

    lookup = build_lookup(objects)
    copied, missing = 0, []
    for raw in filenames:                         # filenames are already “clean”
        src_key = lookup.get(raw)
        if not src_key:
            missing.append(raw)
            continue

        # keep the original relative path under DST_PREFIX
        fname   = Path(src_key).name                         
        dst_key = f"{DST_PREFIX.rstrip('/')}/{fname}"       

        try:
            s3.Object(DST_BUCKET, dst_key).copy_from(
                CopySource={"Bucket": SRC_BUCKET, "Key": src_key}
            )
            logging.info("✔ Copied %s → %s", raw, dst_key)
            copied += 1
        except ClientError as e:
            logging.error("✖ %s: %s", raw, e)
    logging.info("Found %d listed objects.", len(lookup))
    #copy_objects(filenames, lookup)


if __name__ == "__main__":
    main()
