# Running Ray Tune Hyperparameter Optimization on Remote Server

This guide explains how to run the MACD hyperparameter optimization on a remote server with Ray installed.

## Prerequisites

**Remote Server:**
- Ray installed: `pip install ray[tune]`
- All dependencies installed: `pip install -r requirements.txt`
- Data files available in correct location
- MLflow tracking server accessible (optional)

**Local Machine:**
- Ray installed: `pip install ray[tune]`
- Trading codebase with latest changes

## Option 1: SSH and Run Directly (Simplest)

Best for: One-time runs, simple setups

### Steps

1. **Copy codebase to remote server:**
```bash
# Using rsync (recommended)
rsync -avz --exclude='*.pyc' --exclude='__pycache__' \
  E:\Programming\trading_guy/ user@remote-server:/home/user/trading_guy/

# Or using scp
scp -r E:\Programming\trading_guy user@remote-server:/home/user/trading_guy/
```

2. **Ensure data files are present:**
```bash
ssh user@remote-server
cd /home/user/trading_guy
ls -lh data/SPY_UPRO_SPXU_5min.csv  # Verify data file exists
```

3. **Run the optimization:**
```bash
# SSH into remote server
ssh user@remote-server

# Navigate to project
cd /home/user/trading_guy

# Activate virtual environment if needed
source venv/bin/activate

# Run optimization
python trading/backtesting/run_launchers.py

# Or run in background with nohup
nohup python trading/backtesting/run_launchers.py > ray_output.log 2>&1 &

# Monitor progress
tail -f ray_output.log
```

4. **Monitor Ray Dashboard:**
```bash
# Ray dashboard is at http://remote-server-ip:8265
# Port forward if needed:
ssh -L 8265:localhost:8265 user@remote-server
# Then open http://localhost:8265 in your browser
```

## Option 2: Connect to Remote Ray Cluster (Recommended)

Best for: Distributed computing, using multiple machines, continuous experimentation

### Setup Remote Ray Cluster

**On Remote Server:**
```bash
# Start Ray head node
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

# Output will show:
# Ray runtime started.
# To connect to this Ray cluster, use: ray.init("ray://192.168.1.100:10001")
# Dashboard running on http://192.168.1.100:8265

# Copy the ray:// address for use in your code
```

**Add Worker Nodes (Optional):**
```bash
# On additional machines
ray start --address='192.168.1.100:6379'  # Use head node IP
```

### Use Remote Cluster from Local Machine

**Method 1: Use run_launchers_remote.py**

```python
from trading.launchers.run_launchers_remote import run_ray_spy_trend_macd_remote

# Replace with your actual Ray cluster address
best_config = run_ray_spy_trend_macd_remote("ray://192.168.1.100:10001")
```

**Method 2: Modify run_launchers.py**

Add at the beginning of `run_ray_spy_trend_macd()`:
```python
import ray

# Connect to remote cluster before running optimization
ray.init(address="ray://192.168.1.100:10001")

# Rest of function code...
```

### Important Notes for Remote Cluster

1. **Code Availability:**
   - Your code must be available on the remote cluster
   - Ray will serialize Python functions but not the entire codebase
   - Best practice: Clone repo on remote server

2. **Data Availability:**
   - Data files must be accessible from remote cluster
   - Use absolute paths or ensure working directory is correct
   - Consider using shared storage (NFS, S3) for large datasets

3. **MLflow Configuration:**
   - MLflow tracking server should be accessible from remote cluster
   - Use network-accessible tracking URI: `http://hp.lan:8899`
   - Avoid `file://` paths unless using shared filesystem

## Option 3: Ray Job Submission API

Best for: Production deployments, automated workflows

### Submit Job to Remote Cluster

```python
from ray.job_submission import JobSubmissionClient

# Connect to Ray cluster
client = JobSubmissionClient("http://192.168.1.100:8265")

# Submit job
job_id = client.submit_job(
    entrypoint="python trading/backtesting/run_launchers.py",
    runtime_env={
        "working_dir": "./",  # Will upload current directory
        "pip": ["pandas", "numpy", "mlflow"],  # Additional packages
    }
)

print(f"Job submitted: {job_id}")

# Monitor job status
status = client.get_job_status(job_id)
print(f"Job status: {status}")

# Get job logs
logs = client.get_job_logs(job_id)
print(logs)
```

## Configuration for Different Environments

### Local Development
```python
# run_launchers.py (current setup)
# Ray starts automatically, uses local resources
python trading/backtesting/run_launchers.py
```

### Remote Single Server
```python
# On remote server
ray start --head --num-cpus=16 --num-gpus=0
python trading/backtesting/run_launchers.py
```

### Remote Multi-Node Cluster
```bash
# Head node
ray start --head --num-cpus=16

# Worker nodes (on other machines)
ray start --address='head-node-ip:6379' --num-cpus=16

# From local machine
python -c "
import ray
ray.init('ray://head-node-ip:10001')
from trading.backtesting.run_launchers import run_ray_spy_trend_macd
run_ray_spy_trend_macd()
"
```

## Monitoring and Debugging

### Ray Dashboard
- Local: `http://localhost:8265`
- Remote: `http://remote-server-ip:8265`
- Shows: Running trials, resource usage, logs

### MLflow UI
- View results: `http://hp.lan:8899`
- All trials logged automatically
- Compare parameters and metrics

### Ray Status Commands
```bash
# Check cluster status
ray status

# List running jobs
ray list jobs

# Stop a job
ray stop <job_id>

# View logs
ray logs <job_id>
```

## Performance Tuning

### Adjust Concurrent Trials
```python
# In run_ray_spy_trend_macd()
max_concurrent_trials=16,  # Increase based on available CPUs
```

### Resource Allocation
```python
# Specify resources per trial
tune.with_resources(
    trainable,
    resources={"cpu": 2, "gpu": 0}  # 2 CPUs per trial
)
```

### Checkpointing
```python
# Enable checkpointing for long-running experiments
tune.run(
    trainable,
    checkpoint_freq=10,  # Checkpoint every 10 trials
    checkpoint_at_end=True,
)
```

## Troubleshooting

### Issue: "Connection refused"
**Solution:** Ensure Ray head node is running and firewall allows connections on port 10001

### Issue: "Module not found"
**Solution:** Ensure codebase is available on remote cluster or use `runtime_env` to upload code

### Issue: "File not found: data/SPY_UPRO_SPXU_5min.csv"
**Solution:** Use absolute paths or ensure data is copied to remote server

### Issue: Slow performance
**Solution:**
- Check network latency between local machine and remote cluster
- Consider running entirely on remote server (Option 1)
- Increase `max_concurrent_trials` if CPUs are available

### Issue: MLflow logging fails
**Solution:**
- Ensure MLflow server is network-accessible
- Check firewall rules for port 8899
- Use `http://` not `file://` for remote clusters

## Example: Complete Remote Setup

```bash
# === ON REMOTE SERVER ===

# 1. Start Ray cluster
ray start --head --dashboard-host=0.0.0.0

# 2. Clone repository
git clone https://github.com/amperie/trading_guy.git
cd trading_guy

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy data files (if not in repo)
scp user@data-server:/data/SPY_UPRO_SPXU_5min.csv data/

# 5. Verify setup
python -c "import ray; print(ray.available_resources())"


# === ON LOCAL MACHINE ===

# 1. Create remote runner script
cat > run_remote.py << 'EOF'
from trading.backtesting.run_launchers_remote import run_ray_spy_trend_macd_remote

# Run on remote cluster
best_config = run_ray_spy_trend_macd_remote("ray://remote-ip:10001")
print(f"Best config: {best_config}")
EOF

# 2. Run optimization
python run_remote.py

# 3. Monitor in browser
# Ray Dashboard: http://remote-ip:8265
# MLflow UI: http://hp.lan:8899
```

## Summary

**Recommendation:** Use **Option 2 (Remote Ray Cluster)** for best flexibility:
- Run code locally, execute remotely
- Easy monitoring via Ray Dashboard
- Can add/remove workers dynamically
- Full control over resource allocation

**Quick Start:**
1. Remote: `ray start --head --dashboard-host=0.0.0.0`
2. Local: `python trading/backtesting/run_launchers_remote.py`
3. Monitor: Open Ray Dashboard at `http://remote-ip:8265`
