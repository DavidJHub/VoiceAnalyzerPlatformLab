import json
import boto3
from urllib.parse import urlparse
from collections import defaultdict

from database.dbConfig import generate_s3_client



def _add_highlight_error(node):
    """
    Inserta 'highlight_error' cuando encuentra:
      • "mac": true  → "MAC MODULADO/MAC INCORRECTO"
      • "price": true→ "PRECIO MODULADO/PRECIO INCORRECTO"
    Recorre dicts y lists de forma recursiva.
    """
    changed = False

    if isinstance(node, dict):
        if node.get("mac") is True:
            node["highlight_error"] = "MAC MODULADO/MAC INCORRECTO"
            changed = True
        elif node.get("price") is True:
            node["highlight_error"] = "PRECIO MODULADO/PRECIO INCORRECTO"
            changed = True

        for v in node.values():
            if _add_highlight_error(v):
                changed = True

    elif isinstance(node, list):
        for item in node:
            if _add_highlight_error(item):
                changed = True

    return changed

# ────────────────────────────────────────────────────────────────
#  FUNCIÓN PRINCIPAL (RECORRE TODOS LOS SUBDIRECTORIOS)
# ────────────────────────────────────────────────────────────────
def highlight_errors_in_s3_jsons_desc(s3_path: str):
    """
    Procesa TODOS los *.json debajo de `s3_path` empezando por los
    subdirectorios de nombre lexicográficamente descendente.
    """
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    s3_client = generate_s3_client()
    paginator = s3_client.get_paginator("list_objects_v2")

    # 1) Traemos todas las keys
    subdir_map = defaultdict(list)          # {subdir: [keys]}
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.lower().endswith(".json"):
                continue
            # Subdirectorio inmediato tras el prefijo
            remainder = key[len(prefix):]
            subdir = remainder.split("/", 1)[0]  # '202505' en '202505/file.json'
            subdir_map[subdir].append(key)

    # 2) Ordenamos los subdirectorios de Z → A (o 9999 → 0000)
    for subdir in sorted(subdir_map.keys(), reverse=True):
        for key in sorted(subdir_map[subdir]):   # dentro, orden natural opcional
            body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                print(f"⚠️  {key}: JSON no válido, omitido")
                continue

            if _add_highlight_error(data):
                s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=json.dumps(data, ensure_ascii=False, indent=4).encode("utf-8"),
                    ContentType="application/json",
                )
                print(f"✅ {key}: actualizado")
            else:
                print(f"—  {key}: sin cambios")

if __name__ == "__main__":
    highlight_errors_in_s3_jsons_desc("s3://documentos.aihub/Salvador/BancoAgricola/transcript_sentences/")