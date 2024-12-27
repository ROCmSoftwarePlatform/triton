import json
import os


def load_model_config(config_file='model_configs.json'):
    """Load all model configurations from a JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def infer_mnk(model_name, batch_size, seq_len, config_file='model_configs.json'):
    """Infer M, N, and K dimensions for a given model, batch size, and sequence length."""
    configs = load_model_config(config_file)
    if model_name not in configs:
        raise ValueError(f"Model '{model_name}' not found in {config_file}")

    config = configs[model_name]

    # Infer M, N, K based on the feedforward network (FFN) dimensions
    M = batch_size * seq_len  # Total tokens in a batch
    K = config["model_dimension"]  # head dimension * num heads
    N = config["FFN_dimension"]  # size of the intermediate layer of FFN

    return M, N, K


def get_mnk(batch_size=1, seq_len=None, config_file='model_configs.json', model_name=None):
    """
    Retrieve MNK dimensions for benchmarking. Can return:
    - All models (default)
    - A specific model if model_name is provided
    """
    configs = load_model_config(config_file)
    mnk_list = []

    if model_name:
        # Check if the model exists
        if model_name not in configs:
            raise ValueError(f"Model '{model_name}' not found in {config_file}")
        # Handle a specific model
        config = configs[model_name]
        max_ctx_len = config["max_ctx_len"]
        actual_seq_len = max_ctx_len if seq_len is None else seq_len
        M, N, K = infer_mnk(model_name, batch_size, actual_seq_len, config_file)
        mnk_list.append((M, N, K))
    else:
        # Handle all models
        for model_name, config in configs.items():
            max_ctx_len = config["max_ctx_len"]
            actual_seq_len = seq_len or max_ctx_len
            if actual_seq_len > max_ctx_len:
                raise ValueError(f"Sequence length {actual_seq_len} exceeds maximum {max_ctx_len} for {model_name}")
            M, N, K = infer_mnk(model_name, batch_size, actual_seq_len, config_file)
            mnk_list.append((M, N, K))

    return mnk_list


def get_available_models(config_file='model_configs.json'):
    """Load model names from the configuration file."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file)
    with open(config_path, 'r') as f:
        configs = json.load(f)
    return list(configs.keys())


def get_FA_configs(batch_size=1, seq_len=None, model_name=None, config_file='model_configs.json'):
    """
    Retrieve Flash Attention configurations.
    Args:
        batch_size: Batch size for the configurations.
        model_name: Name of the model. If None, return all models.
        config_file: Path to the model configuration file.
    Returns:
        List of Flash Attention configurations as tuples: (BATCH, HQ, HK, N_CTX_Q, N_CTX_K)
    """
    configs = load_model_config(config_file)
    fa_configs = []

    if model_name:
        # Check if the model exists
        if model_name not in configs:
            raise ValueError(f"Model '{model_name}' not found in {config_file}")
        # Handle a specific model
        config = configs[model_name]
        HQ = config["num_attention_heads"]
        HK = HQ if config["num_key_value_heads"] is None else config["num_key_value_heads"]

        max_ctx_len = config["max_ctx_len"]
        N_CTX_Q = N_CTX_K = max_ctx_len if seq_len is None else seq_len
        fa_configs.append((batch_size, HQ, HK, N_CTX_Q, N_CTX_K))
    else:
        # Handle all models
        for model_name, config in configs.items():
            HQ = config["num_attention_heads"]
            HK = HQ if config["num_key_value_heads"] is None else config["num_key_value_heads"]
            max_ctx_len = config["max_ctx_len"]
            N_CTX_Q = N_CTX_K = max_ctx_len if seq_len is None else seq_len
            fa_configs.append((batch_size, HQ, HK, N_CTX_Q, N_CTX_K))

    return fa_configs
