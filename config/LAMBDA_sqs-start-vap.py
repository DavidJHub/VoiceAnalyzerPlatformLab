import boto3
import json

ec2 = boto3.client('ec2', region_name='us-east-1')

def lambda_handler(event, context):
    print("Evento recibido:", json.dumps(event))
    
    response = ec2.start_instances(InstanceIds=['i-07ebde1a635488852'])
    print(f"Instancia {'i-07ebde1a635488852'} iniciada.")
    response = ec2.start_instances(InstanceIds=['i-025d60baf3018d34a'])
    print(f"Instancia {'i-025d60baf3018d34a'} iniciada.")
    response = ec2.start_instances(InstanceIds=['i-0dcb3b2b4309ef0cb'])
    print(f"Instancia {'i-0dcb3b2b4309ef0cb'} iniciada.")
    response = ec2.start_instances(InstanceIds=['i-0bb38733691d343c0'])
    print(f"Instancia {'i-0bb38733691d343c0'} iniciada.")
    return response