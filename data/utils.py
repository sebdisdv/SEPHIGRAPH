from typing import List


def is_static(data: List) -> bool:
    return len(set(data)) == 1
