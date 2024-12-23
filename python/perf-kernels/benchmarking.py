import json


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
    head_count = config["head_count"]
    head_dimension = config["head_dimension"]

    # Infer M, N, K based on the feedforward network (FFN) dimensions
    M = batch_size * seq_len             # Total tokens in a batch
    K = head_dimension * head_count      # Hidden size (d)
    N = 4 * K                            # FFN dimension is typically 4× hidden size

    return M, N, K


def get_all_mnk(batch_size=1, seq_len=None, config_file='model_configs.json'):
    """
    Retrieve all MNK dimensions for benchmarking, iterating over all models in the configuration.
    """
    configs = load_model_config(config_file)
    mnk_list = []

    for model_name, config in configs.items():
        max_seq_len = config["max_seq_len"]
        actual_seq_len = seq_len or max_seq_len  # Use default seq_len if not provided
        if actual_seq_len > max_seq_len:
            raise ValueError(
                f"Sequence length {actual_seq_len} exceeds maximum {max_seq_len} for {model_name}"
            )
        M, N, K = infer_mnk(model_name, batch_size, actual_seq_len, config_file)
        mnk_list.append((M, N, K))

    return mnk_list