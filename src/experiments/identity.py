from __future__ import annotations

import hashlib
from typing import Iterable


def identity_fingerprints(domain: str, identity_column: str, values: Iterable[int]) -> dict[str, str]:
    identity_lines = [
        f"{domain}\t{identity_column}\t{int(value)}\n"
        for value in values
    ]
    return {
        "identity_sha256": hashlib.sha256("".join(sorted(identity_lines)).encode()).hexdigest(),
        "ordered_identity_sha256": hashlib.sha256("".join(identity_lines).encode()).hexdigest(),
    }
