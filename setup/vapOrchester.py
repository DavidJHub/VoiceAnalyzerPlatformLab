import boto3
import json
from datetime import datetime, timedelta, timezone

from database.SQLDataManager import total_files_on_queue
import database.dbConfig as dbcfg

SQS_URL = 'https://sqs.us-east-1.amazonaws.com/891376964786/VAP-Testing'
REGION = 'us-east-1'

# Instancias disponibles 
INSTANCES_BY_LOAD = {
    1: ['i-07ebde1a635488852'],
    2: ['i-07ebde1a635488852', 'i-025d60baf3018d34a'],
    3: ['i-07ebde1a635488852', 'i-025d60baf3018d34a', 'i-0dcb3b2b4309ef0cb'],
    4: ['i-07ebde1a635488852', 'i-025d60baf3018d34a', 'i-0dcb3b2b4309ef0cb', 'i-0bb38733691d343c0'],
}



def determine_instance_count(file_count):
    if file_count < 800:
        return 1
    elif file_count < 1600:
        return 2
    elif file_count < 2400:
        return 3
    else:
        return 4

def send_start_signal(sqs_url, instance_ids, region):
    sqs = boto3.client('sqs', region_name=region)
    message = {
        "action": "start",
        "instances": instance_ids
    }
    response = sqs.send_message(
        QueueUrl=sqs_url,
        MessageBody=json.dumps(message)
    )
    print("Mensaje enviado a SQS:", response['MessageId'])

def main():
    conn = dbcfg.conectar(HOST=dbcfg.HOST_DB_VAP,  
                                DATABASE=dbcfg.DB_NAME_VAP,  
                                USERNAME=dbcfg.USER_DB_VAP,  
                                PASSWORD=dbcfg.PASSWORD_DB_VAP)
    file_count = total_files_on_queue(conn)
    print(f"Archivos cargados en las últimas 24 horas: {file_count}")

    instance_count = determine_instance_count(file_count)
    instance_ids = INSTANCES_BY_LOAD[instance_count]
    print(f"Encendiendo {instance_count} instancia(s): {instance_ids}")

    send_start_signal(SQS_URL, instance_ids, REGION)

if __name__ == "__main__":
    main()
