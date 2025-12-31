from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import dataclasses
import hashlib
import json


def fingerprint(cfg: "Config") -> str:
  """Stable config fingerprint for reproducible resume checks.

  Uses a canonical JSON encoding of the dataclass fields (sorted keys) and
  returns a SHA-256 hex digest.
  """
  d = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else {"value": str(cfg)}
  d = {k: v for k, v in d.items() if not str(k).startswith("_")}
  s = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
  return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class Config:
  # =============================================================================
  # Meta
  # =============================================================================
  preset: str = "custom"
  experiment_id: str = "default"

  # =============================================================================
  # Model Architecture (REQUIRED - no defaults)
  # =============================================================================
  vocab_size: int = 201088  # o200k_harmony tokenizer
  tokenizer: str = "o200k_harmony"  # Tokenizer name for data prep
  eos_token_id: int = 199999  # o200k_harmony eot_token; also used as padding
  dim: int = None  # Must specify
  n_layers: int = None  # Must specify
  n_heads: int = None  # Must specify

  # Model Architecture (MoE - required for MoE models)
  inter_dim: int = None  # Dense MLP intermediate dim
  moe_inter_dim: int = None  # Expert MLP intermediate dim
  n_routed_experts: int = None  # Total routed experts
  n_activated_experts: int = None  # Top-K routing
  n_shared_experts: int = 2  # Shared experts (default 2)
  n_dense_layers: int = 1  # First N layers are dense

  # Attention (MLA defaults)
  attn: str = "mla"  # Global attention type
  attn_local: str = "swa"  # Local attention type (used when attn_global_every > 1)
  attn_global_every: int = 1  # Every Nth layer is global; 1 = all global
  attn_local_window: int = 128  # Window size for local attention layers
  q_lora_rank: int = 1536
  kv_lora_rank: int = 512
  qk_nope_head_dim: int = 128
  qk_rope_head_dim: int = 64
  v_head_dim: int = 128

  # RoPE
  max_position_embeddings: int = 8192
  rope_theta: float = 50000.0
  rope_scaling_factor: float = 1.0
  rope_ntk_alpha: float = 1.0
  rope_ntk_beta: float = 32.0

  # Normalization
  rms_norm_eps: float = 1e-5

  # MoE Routing
  router_bias_update_rate: float = 1e-4
  aux_loss_alpha: float = 0.0
  norm_topk_prob: bool = True
  route_scale: float = 1.0

  # Precision
  dtype: Optional[str] = "bf16"  # bf16 | fp8 | nvfp4

  # =============================================================================
  # Optimizer
  # =============================================================================
  lr_dense: float = 3.4e-4
  lr_router: float = 3.4e-4
  lr_expert: float = 3.4e-4
  weight_decay: float = 0.1
  adam_beta1: float = 0.9
  adam_beta2: float = 0.95
  adam_beta2_expert: float = 0.99  # Higher for FP8 gradient noise
  adam_eps: float = 1e-8
  muon_momentum: float = 0.95

  # =============================================================================
  # Scheduler (WSD - Warmup-Sustain-Decay)
  # =============================================================================
  warmup_steps: int = 500
  hold_tokens: int = 10_000_000_000  # 10B tokens sustain
  decay_tokens: int = 40_000_000_000  # 40B tokens decay
  decay_floor: float = 3e-5  # Absolute LR floor (not a ratio)

  # =============================================================================
  # Training
  # =============================================================================
  steps: int = 10000
  batch_size: int = 8
  seq_len: int = 4096
  seed: int = 42
  log_every: int = 10

  # Checkpointing
  checkpoint_dir: str = "/data/checkpoints"
  checkpoint_every: int = 1000
  checkpoint_keep_last_n: int = 3
  resume: bool = True

  # =============================================================================
  # Data
  # =============================================================================
  data_root: str = "/data"
  data_path: Optional[str] = None  # Direct path to .npy shards (or cache location for HF)

  # HuggingFace datasets (auto-download and cache)
  hf_dataset: Optional[str] = None  # e.g., "HuggingFaceFW/fineweb-edu"
  hf_split: str = "train"
  hf_subset: Optional[str] = None  # Dataset subset/config name
  hf_text_field: str = "text"  # Field containing text
  hf_data_files: Optional[str] = None  # Glob pattern for specific files

  # Flows (power-user multi-stage training)
  mixture_toml: Optional[str] = None
  flow_profiles_toml: Optional[str] = None
  flow_mode: Optional[str] = None  # dev | test | ablation | proxy | full_train
  active_params_b: Optional[float] = None  # Active params in billions (for flow budgets)

  # =============================================================================
  # Metrics & Experiment Tracking
  # =============================================================================
  metrics_dir: str = "/data/metrics"
  experiments_db: str = "/data/experiments.db"

  # =============================================================================
  # SFT (optional)
  # =============================================================================
  sft_enabled: bool = False
  sft_prompt_format: str = "chatml"  # chatml | llama3 | custom
  sft_mask_prompt_loss: bool = True
  sft_packing_enabled: bool = False
  sft_data_path: Optional[str] = None

  # =============================================================================
  # RL (optional)
  # =============================================================================
  rl_enabled: bool = False
  rl_algorithm: str = "grpo"  # grpo | ppo | dpo
  grpo_kl_beta: float = 0.001
  grpo_clamp_eps_lower: float = 0.01
  grpo_clamp_eps_upper: float = 0.01
  grpo_group_size: int = 2
  grpo_prompts_per_step: int = 32
  grpo_iterations: int = 2
  grpo_temperature: float = 1.0
  grpo_top_p: float = 0.9
  reward_model_path: Optional[str] = None

  # =============================================================================
  # Validation (optional)
  # =============================================================================
  validation_enabled: bool = False
  validation_data_path: Optional[str] = None
  validation_every: int = 500
  validation_steps: int = 100

  # =============================================================================
  # Profiling (optional)
  # =============================================================================
  profiling_enabled: bool = False
  profiling_steps: int = 10
  profiling_output_dir: str = "/data/profiles"

  # =============================================================================
  # Evaluation (optional)
  # =============================================================================
  eval_enabled: bool = False
  eval_mode: str = "reserved_gpu"  # inline | reserved_gpu | k8s_job
  eval_every: int = 0               # steps; 0 disables
  eval_tasks: str = "core"         # optional shorthand; prefer tasks file
  eval_tasks_file: str = "configs/eval/tasks.toml"
  eval_budget_max_examples: int = 500
  eval_budget_max_time_s: int = 300
  eval_reserved_gpu_id: int = 7

  # =============================================================================
  # Backend-specific nested configs (stored as dicts)
  # =============================================================================
  attn_swa: Dict[str, Any] = field(default_factory=dict)
  attn_nsa: Dict[str, Any] = field(default_factory=dict)
  attn_dsa: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Config attributes used by nmoe.attention.*
# =============================================================================
#
# Core Dimensions:
#   dim                 - Model/hidden dimension (MLA, SWA, NSA, DSA, KDA)
#
# Head Configuration:
#   n_heads             - Number of attention heads (MLA, SWA, NSA, DSA, KDA)
#
# LoRA Ranks:
#   q_lora_rank         - Query LoRA rank (MLA, DSA, KDA)
#   kv_lora_rank        - Key-Value LoRA rank (MLA, DSA, KDA)
#
# Head Dimensions:
#   qk_nope_head_dim    - Q/K head dim without RoPE (MLA, SWA, NSA, DSA, KDA)
#   qk_rope_head_dim    - Q/K head dim with RoPE (MLA, SWA, NSA, DSA, KDA)
#   v_head_dim          - Value head dimension (MLA, SWA, NSA, DSA, KDA)
#
# Normalization:
#   rms_norm_eps        - RMSNorm epsilon (MLA, DSA, KDA)
#
# Backend-Specific Options:
#
#   attn_swa (dict):
#     kv_heads          - Number of KV heads for GQA (default: n_heads)
#
#   attn_nsa (dict):
#     kv_heads          - Number of KV heads (default: attn_swa.kv_heads or n_heads)
#     cmp_len           - Compression block length (default: 32)
#     cmp_stride        - Compression stride (default: 16)
#     slc_block         - Selection block size (default: 64)
#     topk_blocks       - Number of selected blocks (default: 16)
#     tile_q            - Query tiling parameter (default: 128)
#     flex_block        - FlexAttention block size (default: 128)
#     cmp_hidden        - Compression MLP hidden dim (default: v_head_dim * 2)
#     gate_hidden       - Gate MLP hidden dim (default: dim // 4)
#
#   attn_dsa (dict):
#     n_idx_heads       - Number of indexer heads (default: 4)
#     idx_dim           - Indexer dimension (default: 64)
#     top_k             - Top-k tokens to select (default: 2048)
#     training_mode     - 'dense_warmup' or 'sparse' (default: 'sparse')
#
# =============================================================================
# Config attributes used by nmoe.model
# =============================================================================
#
# Model Architecture:
#   dim                   - Model/hidden dimension
#   inter_dim             - MLP intermediate dimension
#   n_layers              - Number of transformer blocks
#   n_dense_layers        - Number of dense (non-MoE) layers at start
#   vocab_size            - Vocabulary size
#
# Attention:
#   attn                  - Attention backend: 'mla', 'swa', 'nsa', 'dsa', 'kda'
#   qk_rope_head_dim      - RoPE head dimension (for RotaryEmbedding)
#
# RoPE:
#   rope_theta            - RoPE base frequency
#   rope_scaling_factor   - RoPE scaling factor (YaRN)
#   rope_ntk_alpha        - NTK-aware scaling alpha
#   rope_ntk_beta         - NTK-aware scaling beta
#   max_position_embeddings - Maximum sequence length
#
# Normalization:
#   rms_norm_eps          - RMSNorm epsilon
#
# MoE / Router:
#   n_routed_experts      - Total number of routed experts
#   n_activated_experts   - Number of experts activated per token (topk)
#   n_shared_experts      - Number of shared experts (fused into single MLP)
#   moe_inter_dim         - MoE expert intermediate dim (default: inter_dim)
#   route_scale           - Multiplies weights AFTER normalization (default: 1.0)
#
# Optional:
#   dtype                 - 'bf16', 'fp8', or 'nvfp4' (default: 'bf16')


# =============================================================================
# TOML Config Loading with Environment Variable Expansion
# =============================================================================

import os
import re
import tomllib
from pathlib import Path
from typing import Any, Union

# Allowlisted env var prefixes (security: prevent accidental secret leakage)
# NOTE: HF_ intentionally excluded (HF_TOKEN is sensitive)
_ENV_VAR_PREFIXES = ("NMOE_", "HYDRA_")


class ConfigEnvError(Exception):
    """Raised when env var expansion fails."""
    pass


def _expand_env_vars(obj: Any, source: str = "<config>") -> Any:
    """Recursively expand ${VAR} and ${VAR:-default} in strings.

    Args:
        obj: Config object (dict, list, or scalar)
        source: Source file path for error messages

    Returns:
        Object with env vars expanded

    Raises:
        ConfigEnvError: If env var is not in allowlist or unresolved
    """
    if isinstance(obj, str):
        def replace(m: re.Match) -> str:
            var = m.group(1)
            default = m.group(3)  # None if no default

            # Security: only allow specific prefixes
            if not any(var.startswith(p) for p in _ENV_VAR_PREFIXES):
                raise ConfigEnvError(
                    f"Env var ${{{var}}} in {source} not allowed. "
                    f"Only prefixes {_ENV_VAR_PREFIXES} are permitted."
                )

            value = os.environ.get(var)
            if value is not None:
                return value
            if default is not None:
                return default
            raise ConfigEnvError(
                f"Unresolved env var ${{{var}}} in {source}. "
                f"Set {var} or provide default: ${{{var}:-/default/path}}"
            )

        return re.sub(r'\$\{([A-Z_][A-Z0-9_]*)(:-([^}]*))?\}', replace, obj)

    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v, source) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [_expand_env_vars(v, source) for v in obj]

    return obj


def _check_unresolved(obj: Any, source: str = "<config>") -> None:
    """Fail-fast if any ${...} patterns remain after expansion."""
    if isinstance(obj, str):
        if "${" in obj:
            match = re.search(r'\$\{[^}]+\}', obj)
            if match:
                raise ConfigEnvError(
                    f"Unresolved placeholder {match.group(0)} in {source}. "
                    f"This may indicate a malformed env var reference."
                )
    elif isinstance(obj, dict):
        for v in obj.values():
            _check_unresolved(v, source)
    elif isinstance(obj, list):
        for v in obj:
            _check_unresolved(v, source)


def load_toml(path: Union[str, Path]) -> dict:
    """Load TOML config with env var expansion.

    Supports ${VAR} and ${VAR:-default} syntax.
    Only NMOE_ and HYDRA_ prefixed vars are allowed.

    Args:
        path: Path to TOML file

    Returns:
        Parsed and expanded config dict

    Raises:
        ConfigEnvError: If env var not allowed or unresolved
    """
    path = Path(path)
    with open(path, "rb") as f:
        obj = tomllib.load(f)

    obj = _expand_env_vars(obj, str(path))
    _check_unresolved(obj, str(path))
    return obj
