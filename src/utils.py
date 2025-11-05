import os
import re


def stem(name: str) -> str:
    base = os.path.basename(name)
    base = re.sub(r"(\.(png|jpg|jpeg|npy))$", "", base)
    base = base.replace("_depth", "").replace("_mask", "")
    return base
