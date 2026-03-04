#!/usr/bin/env python3
import json, os, argparse
from datetime import datetime, timezone
import urllib.request, urllib.error
import boto3

DEFAULT_REGION = os.getenv("AWS_REGION", "us-east-1")

def get_instance_id():
    """Obtiene el ID de la instancia usando IMDSv2 con fallback a IMDSv1."""
    # Intento IMDSv2
    try:
        req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        )
        token = urllib.request.urlopen(req, timeout=2).read().decode()
        req2 = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-id",
            headers={"X-aws-ec2-metadata-token": token},
        )
        iid = urllib.request.urlopen(req2, timeout=2).read().decode()
        return iid
    except Exception:
        # Fallback IMDSv1
        try:
            iid = urllib.request.urlopen(
                "http://169.254.169.254/latest/meta-data/instance-id", timeout=2
            ).read().decode()
            return iid
        except Exception as e:
            raise RuntimeError(f"No se pudo obtener instance-id: {e}")

def send_stop_signal(queue_url: str, region: str, instances, extra=None):
    sqs = boto3.client("sqs", region_name=region)
    payload = {
        "action": "stop",
        "instances": instances,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    resp = sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(payload))
    print(f"[OK] Mensaje enviado a SQS: {resp['MessageId']} -> {payload}")

def main():
    parser = argparse.ArgumentParser(description="Enviar señal de apagado por SQS")
    parser.add_argument("--queue-url", required=True, help="URL de la cola SQS destino")
    parser.add_argument("--region", default=DEFAULT_REGION, help="Región AWS (ej. us-east-1)")
    parser.add_argument("--include-self", action="store_true",
                        help="Apagar esta misma instancia (por defecto sí)")
    parser.add_argument("--also", nargs="*", default=[],
                        help="IDs extra de instancias a apagar (opcional)")
    parser.add_argument("--job-id", default=None, help="ID del job para trazabilidad (opcional)")
    args = parser.parse_args()

    instances = []
    if args.include_self or args.include_self is False:  # por claridad
        instances.append(get_instance_id())
    instances.extend(args.also)

    if not instances:
        raise SystemExit("No hay instancias para apagar.")

    extra = {"job_id": args.job_id} if args.job_id else None
    send_stop_signal(args.queue_url, args.region, instances, extra=extra)

if __name__ == "__main__":
    main()
