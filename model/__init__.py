from .gru import GRUModel
from .lstm import LSTMModel
from .rnn import RNNModel
from .transformer import TransformerModel

MODELS = {"rnn": RNNModel, "gru": GRUModel, "lstm": LSTMModel, "transformer": TransformerModel}


def build_model(cfg: dict, vocab_size: int):
    return MODELS[cfg["type"]](vocab_size=vocab_size, **cfg.get("args", {}))
