from .rnn import RNNModel

MODELS = {"rnn": RNNModel}


def build_model(cfg: dict, vocab_size: int):
    return MODELS[cfg["type"]](vocab_size=vocab_size, **cfg.get("args", {}))
