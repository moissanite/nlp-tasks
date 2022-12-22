from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    data_dir: str
    trim_padding: bool
    labels: list


@dataclass(frozen=True)
class Model:
    name: str
    dropout: float
    max_len: int
    save_dir: str
    save_best: str
    save_last: str


@dataclass(frozen=True)
class Train:
    seed: int
    batch_size: int
    epochs: int
    lr: float
    eps: float


@dataclass(frozen=True)
class Result:
    dir: str


class TrainConfig:
    def __init__(self, **kwargs):
        self.dataset = Dataset(**kwargs['dataset'])
        self.model = Model(**kwargs['model'])
        self.train = Train(**kwargs['train'])
        self.result = Result(**kwargs['result'])
