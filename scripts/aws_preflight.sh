#!/usr/bin/env bash
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI is not installed. Install AWS CLI v2, then run this again." >&2
  exit 1
fi

echo "AWS identity:"
aws sts get-caller-identity

echo
echo "Region: $REGION"
aws configure get region || true

echo
echo "Checking EC2 G instance vCPU quota."
aws service-quotas get-service-quota \
  --region "$REGION" \
  --service-code ec2 \
  --quota-code L-DB2E81BA \
  --query 'Quota.{Name:QuotaName,Value:Value}' \
  --output table || true

echo
echo "Checking for recent Deep Learning GPU AMIs."
aws ec2 describe-images \
  --region "$REGION" \
  --owners amazon \
  --filters 'Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu*' 'Name=state,Values=available' \
  --query 'sort_by(Images,&CreationDate)[-5:].{Name:Name,ImageId:ImageId,CreationDate:CreationDate}' \
  --output table
