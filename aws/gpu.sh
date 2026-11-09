#!/usr/bin/env bash
# Minimal cost-guarded GPU instance manager. Usage: bash aws/gpu.sh {launch|stop|start|kill|status|ssh}
set -euo pipefail

# ---- EDIT THESE ----
KEY_NAME="mlchal"            # EC2 key pair name you created
KEY_FILE="$HOME/mlchal.pem"  # path to the downloaded .pem
INSTANCE_TYPE="g5.xlarge"    # A10G 24GB ~ $1.0-1.4/hr in ap-south-1
REGION="ap-south-1"          # Mumbai
# --------------------

TAG="mlchallenge2026"
AMI=$(aws ec2 describe-images --region "$REGION" --owners amazon \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu*" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text 2>/dev/null || echo "")

iid() { aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:Name,Values=$TAG" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text; }

ip() { aws ec2 describe-instances --region "$REGION" --instance-ids "$1" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text; }

case "${1:-status}" in
  launch)
    [ "$(iid)" != "None" ] && { echo "already exists: $(iid) — use start/status"; exit 1; }
    ID=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
      --key-name "$KEY_NAME" \
      --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":150,"VolumeType":"gp3"}}]' \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG}]" \
      --query 'Instances[0].InstanceId' --output text)
    echo "launched $ID — waiting for IP..."
    aws ec2 wait instance-running --region "$REGION" --instance-ids "$ID"
    echo "SSH:  ssh -i $KEY_FILE ubuntu@$(ip "$ID")"
    echo "REMEMBER: bash aws/gpu.sh stop  when you step away. Meter is running."
    ;;
  stop)   aws ec2 stop-instances  --region "$REGION" --instance-ids "$(iid)" >/dev/null && echo "stopping (billing ~paused)";;
  start)  aws ec2 start-instances --region "$REGION" --instance-ids "$(iid)" >/dev/null \
          && aws ec2 wait instance-running --region "$REGION" --instance-ids "$(iid)" \
          && echo "SSH:  ssh -i $KEY_FILE ubuntu@$(ip "$(iid)")";;
  kill)   aws ec2 terminate-instances --region "$REGION" --instance-ids "$(iid)" >/dev/null && echo "terminated (final)";;
  ssh)    exec ssh -i "$KEY_FILE" "ubuntu@$(ip "$(iid)")";;
  status) aws ec2 describe-instances --region "$REGION" --filters "Name=tag:Name,Values=$TAG" \
            --query 'Reservations[].Instances[].{id:InstanceId,state:State.Name,type:InstanceType,ip:PublicIpAddress,launched:LaunchTime}' --output table;;
  *) echo "usage: bash aws/gpu.sh {launch|stop|start|kill|status|ssh}";;
esac
