from typing import Mapping
from archs.cnnocr import CNNOCR


def get_seq_len(state_dict: Mapping[str, object], seq_key: str) -> int:
    # Определяет длину последовательности по ключам вида "{seq_key}.{i}.{suffix}"
    prefix = seq_key + "."
    keys = set()
    for k in state_dict.keys():
        if k.startswith(prefix):
            index = k[len(prefix) :].split(".", maxsplit=1)[0]
            keys.add(int(index))
    return max(keys) + 1 if keys else 0


def parse_model(state_dict):
    # Извлекаем параметры из state_dict и создаем модель CNNOCR
    in_ch = state_dict["feature.stages.0.down.weight"].shape[1]
    dims = [state_dict[f"feature.stages.{i}.down.weight"].shape[0] for i in range(4)]
    depths = [get_seq_len(state_dict, f"feature.stages.{i}.body") for i in range(4)]
    hidden_size = state_dict["SequenceModeling.0.linear.weight"].shape[0]
    num_classes = state_dict["fc.weight"].shape[0]
    model = CNNOCR(num_classes, hidden_size, in_ch, depths, dims, 0)
    model.load_state_dict(state_dict)
    return model, in_ch == 3
