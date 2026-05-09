# LogSage AWS EC2 Runbook

This runbook trains LogSage on a fresh AWS GPU EC2 instance and uploads the LoRA adapter to Hugging Face.

## Local Preflight

```bash
cd /home/lenovo/Documents/logsage
./venv/bin/python validate_dataset.py
./venv/bin/python -m py_compile logsage_data.py validate_dataset.py train_logsage.py test_logsage.py
./venv/bin/hf auth whoami
```

Expected Hugging Face user: `auro-rirum`.

AWS CLI is not installed on this machine yet. After installing AWS CLI v2 and running `aws configure`, check readiness with:

```bash
cd /home/lenovo/Documents/logsage
chmod +x scripts/aws_preflight.sh
AWS_REGION=us-east-1 ./scripts/aws_preflight.sh
```

## AWS Defaults

- Region: `us-east-1`
- Preferred instance: `g5.2xlarge`
- Fallback instance: `g5.xlarge`
- Disk: 100 GB gp3
- AMI: AWS Deep Learning GPU AMI with PyTorch/CUDA
- Security group: SSH from your current public IP only
- Budget guardrails: AWS Budget alerts at `$55` and `$65`, plus the instance-side shutdown timer in `scripts/ec2_train.sh`

Before launch, confirm the EC2 quota named `Running On-Demand G and VT instances` is high enough for `g5.2xlarge` or `g5.xlarge`.

## Upload Project To EC2

Exclude the local virtualenv because it is large and machine-specific.

```bash
rsync -av --exclude venv --exclude runs --exclude LogSage-Qwen2.5-7B-QLoRA-v0 \
  /home/lenovo/Documents/logsage/ ubuntu@EC2_PUBLIC_IP:~/logsage/
```

If the AMI uses `ec2-user`, replace `ubuntu` with `ec2-user`.

## Train On EC2

SSH into the instance, then run:

```bash
cd ~/logsage
chmod +x scripts/ec2_train.sh
export HF_TOKEN="your_hugging_face_write_token"
export HF_REPO_ID="auro-rirum/LogSage-Qwen2.5-7B-QLoRA-v0"
./scripts/ec2_train.sh
```

The script performs:

- 8-hour safety shutdown
- GPU check with `nvidia-smi`
- clean virtualenv creation
- dependency install
- dataset validation
- 10-example smoke training run
- full QLoRA training
- Hugging Face adapter upload
- sample inference test

## Watch Training

In a second SSH terminal:

```bash
cd ~/logsage
tail -f runs/*/train.log
```

For TensorBoard:

```bash
cd ~/logsage
source venv/bin/activate
tensorboard --logdir runs --host 0.0.0.0 --port 6006
```

Then open an SSH tunnel from the local machine:

```bash
ssh -L 6006:localhost:6006 ubuntu@EC2_PUBLIC_IP
```

Open `http://localhost:6006`.

## Retrieve Artifacts

The trainer uploads to Hugging Face directly, but you can also copy local artifacts back:

```bash
rsync -av ubuntu@EC2_PUBLIC_IP:~/logsage/LogSage-Qwen2.5-7B-QLoRA-v0/ \
  /home/lenovo/Documents/logsage/LogSage-Qwen2.5-7B-QLoRA-v0/
rsync -av ubuntu@EC2_PUBLIC_IP:~/logsage/runs/ \
  /home/lenovo/Documents/logsage/runs/
```

## Cleanup

After confirming the Hugging Face repo has the adapter and report:

```bash
sudo shutdown -h now
```

Then terminate the EC2 instance and delete any unattached EBS volumes from the AWS console.
