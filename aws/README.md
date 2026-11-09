# AWS: spend the $200 only on runs that matter

Rules of thumb
- Free first: Kaggle (2×T4, ~30 GPU-h/week/account) and Colab for EDA + samples.
- Paid (g5.xlarge, A10G 24GB, ~$1.0-1.4/hr) only for full-data training, Day 2-3.
- NEVER leave an instance running idle. `stop` when stepping away (disk persists, billing ~stops); `terminate` when done for good.
- Set a billing alarm the moment credits land: Billing > Budgets > create $150 budget with email alert.

One-time setup (5 min, do before Thursday)
1. Install AWS CLI v2, run `aws configure` (create an access key: IAM > your user > Security credentials).
2. Create a key pair: EC2 > Key Pairs > create `mlchal` (download .pem).
3. Request quota for "All G and VT Spot/On-Demand instances" >= 4 vCPUs if 0 (Service Quotas > EC2). Do this NOW - approval can take hours.

Scripts (edit the 3 vars at the top of gpu.sh first)
- `bash aws/gpu.sh launch`  -> starts a g5.xlarge with the Deep Learning AMI, prints SSH command
- `bash aws/gpu.sh stop`    -> stops it (cheap: only disk ~$0.08/GB-month)
- `bash aws/gpu.sh start`   -> resumes it
- `bash aws/gpu.sh kill`    -> terminates it (final)
- `bash aws/gpu.sh status`  -> state + uptime = money meter

On the instance: `git clone` the kit repo, download the dataset DIRECTLY on the instance
(fast AWS network - never through home wifi), `pip install -r requirements.txt`, run in
`tmux` so SSH drops don't kill training. Copy small outputs back with `scp` (preds only).

SageMaker Studio Lab / notebooks are an alternative UI but the EC2 route above is the
cheapest per credit and what the scripts assume.
