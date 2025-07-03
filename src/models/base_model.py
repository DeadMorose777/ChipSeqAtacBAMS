import torch, abc
class BaseModel(torch.nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, batch: dict) -> torch.Tensor: ...

    @abc.abstractmethod
    def collate_fn(self, samples: list[dict]) -> dict: ...
