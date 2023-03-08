import torch


def detect_device(whitelist=None):
    whitelist = whitelist if whitelist is not None else [
        "cuda",
        "mps",
        "cpu"
    ]
    device_mapping = {
        "cuda": torch.cuda,
        "mps": torch.backends.mps,
    }
    for device in whitelist:
        if device == "cpu" or device_mapping[device].is_available():
            return device
    else:
        raise Exception("No available device!")
