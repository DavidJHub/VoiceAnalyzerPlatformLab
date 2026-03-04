import database.dbConfig as dbcfg

def get_s3_client():
    return dbcfg.generate_s3_client()


def delete_concat_objects(bucket_name: str, prefix: str | None = None, dry_run: bool = False) -> None:
    """
    Borra todos los objetos de un bucket S3 cuyo key contenga '-concat'.

    :param bucket_name: Nombre del bucket de S3.
    :param prefix: (Opcional) Prefijo para limitar la búsqueda (por ejemplo 'process/').
    :param dry_run: Si es True, solo muestra qué borraría, sin borrar nada.
    """
    s3 = get_s3_client()
    paginator = s3.get_paginator('list_objects_v2')

    list_kwargs = {"Bucket": bucket_name}
    if prefix:
        list_kwargs["Prefix"] = prefix

    total_encontrados = 0
    total_borrados = 0

    for page in paginator.paginate(**list_kwargs):
        contents = page.get("Contents", [])
        if not contents:
            continue

        # Filtrar solo los que tienen '-concat' en el key
        keys_to_delete = [
            {"Key": obj["Key"]}
            for obj in contents
            if "-concat" in obj["Key"]
        ]

        if not keys_to_delete:
            continue

        total_encontrados += len(keys_to_delete)
        print("Encontrados (en esta página):")
        for k in keys_to_delete:
            print("  -", k["Key"])

        if dry_run:
            # Solo mostrar, no borrar
            continue

        # Borrar por lotes de máximo 1000 objetos
        BATCH_SIZE = 1000
        for i in range(0, len(keys_to_delete), BATCH_SIZE):
            batch = keys_to_delete[i : i + BATCH_SIZE]
            response = s3.delete_objects(
                Bucket=bucket_name,
                Delete={
                    "Objects": batch,
                    # OJO: sin Quiet=True para ver bien la respuesta
                }
            )

            # Mostrar respuesta cruda por si hay errores
            print("Respuesta delete_objects:")
            print(json.dumps(response, indent=2, default=str))

            borrados = len(response.get("Deleted", []))
            total_borrados += borrados
            print(f"Eliminados {borrados} objetos en este lote.")

            errors = response.get("Errors", [])
            if errors:
                print("⚠️ Errores al borrar:")
                for e in errors:
                    print(f"  Key={e.get('Key')} Code={e.get('Code')} Message={e.get('Message')}")

    print(f"\nObjetos encontrados con '-concat': {total_encontrados}")
    print(f"Objetos borrados: {total_borrados}")



if __name__ == "__main__":
    # Ejemplos de uso:
    # Solo bucket:
    # delete_concat_objects("mi-bucket")

    # Bucket + prefijo, si quieres limitar a una carpeta:
    delete_concat_objects("s3iahub.igs", prefix="Colombia/Bancolombia/2025-12-04/")

    pass
