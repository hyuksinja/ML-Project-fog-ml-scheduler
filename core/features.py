"""
core/features.py
────────────────
Heterogeneous-Context Feature Engineering (HCFE)  — Novel Contribution 1

Defines the 18 features spanning three orthogonal dimensions:
  • Task dimension       (6 raw task attributes)
  • Node dimension       (5 raw node attributes)
  • Interaction dimension(7 derived cross-features — the novel part)

The interaction features encode domain knowledge that no prior fog-cloud
scheduling paper combines in a single feature vector, which forms a
key patentability argument.
"""

# ── Raw feature names ────────────────────────────────────────────────
TASK_FEATURES = [
    "task_size_mi",       # Task size in MIPS instructions
    "task_mem_req_mb",    # Memory requirement in MB
    "task_data_size_mb",  # Input/output data size in MB
    "task_priority",      # QoS priority level (1–3)
    "task_deadline_s",    # Soft deadline in seconds
    "task_type_id",       # Encoded task category
]

NODE_FEATURES = [
    "node_mips",              # Processing capacity (MIPS)
    "node_ram_gb",            # Available RAM in GB
    "node_bandwidth_mbps",    # Network bandwidth in Mbps
    "node_load",              # Current CPU utilisation (0–1)
    "node_type",              # Node tier: 0=fog-edge, 1=fog-mid, 2=cloud
]

# ── Novel interaction / derived features (HCFE contribution) ─────────
INTERACTION_FEATURES = [
    "cpu_demand_ratio",    # task_size_mi / node_mips
    "mem_pressure",        # task_mem_req_mb / (node_ram_gb * 1024)
    "net_intensity",       # task_data_size_mb / node_bandwidth_mbps
    "effective_mips",      # node_mips * (1 - node_load)
    "deadline_slack",      # task_deadline_s / estimated_base_exec_time
    "load_mem_interact",   # node_load × task_mem_req_mb  (interaction term)
    "tier_task_match",     # 1 if node tier aligns with task type class
]

# ── Combined feature list (used everywhere in training / inference) ───
FEATURE_COLS = TASK_FEATURES + NODE_FEATURES + INTERACTION_FEATURES

TARGET_COL = "execution_time_s"

__all__ = [
    "TASK_FEATURES",
    "NODE_FEATURES",
    "INTERACTION_FEATURES",
    "FEATURE_COLS",
    "TARGET_COL",
]
