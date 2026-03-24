import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate features
task_size = np.random.uniform(10, 5000, n_samples)  # MB, size of the task
memory_required = np.random.uniform(256, 8192, n_samples)  # MB
cpu_speed = np.random.uniform(1.0, 3.5, n_samples)  # GHz
node_load = np.random.uniform(0.1, 0.9, n_samples)  # [0, 1], normalized load
available_memory = np.random.uniform(512, 32768, n_samples)  # MB
network_latency = np.random.uniform(5, 200, n_samples)  # ms

# Realistic formula for execution time (in seconds)
# Factors:
# - Larger tasks and memory needs increase time.
# - Higher CPU speed and more available memory decrease time.
# - Higher load and latency increase time.
# Add noise for realism.
execution_time = (
    0.4 * (task_size / (cpu_speed * 1000)) +     # Scaled by CPU speed
    0.2 * (memory_required / np.maximum(available_memory, 1)) * 10 +  # Memory bottleneck
    0.25 * (node_load * task_size / 1000) +      # Node load penalty
    0.15 * (network_latency / 20) +              # Network delay
    np.random.normal(0, 0.8, n_samples)          # Noise
)

execution_time = np.clip(execution_time, a_min=0.2, a_max=None)  # Min exec time 0.2s

# Assemble dataframe
df = pd.DataFrame({
    "task_size": task_size,
    "memory_required": memory_required,
    "cpu_speed": cpu_speed,
    "node_load": node_load,
    "available_memory": available_memory,
    "network_latency": network_latency,
    "execution_time": execution_time
})

# Save to CSV
df.to_csv("dataset.csv", index=False)