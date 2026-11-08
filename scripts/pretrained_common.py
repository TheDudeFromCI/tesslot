"""Shared runtime-plan construction for local pretrained CUDA probes."""


def runtime_plan(config, parameter_budget_bytes):
    layers = config['num_hidden_layers']
    token = config['bos_token_id']
    if not isinstance(layers, int) or not 1 <= layers <= 1024 or not isinstance(token, int) or not 0 <= token < config['vocab_size']:
        raise ValueError('invalid layer count or BOS ID')
    target = dict(kind='cuda', index=0)
    return dict(root=target, layers=[target.copy() for _ in range(layers)], precision='tf32', parameter_budget_bytes=parameter_budget_bytes)
