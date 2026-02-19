from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional


@dataclass
class StreamSimulator:
    paths: List[Path]
    limit: Optional[int] = None

    def __iter__(self) -> Iterator[Path]:
        count = 0
        for p in self.paths:
            yield p
            count += 1
            if self.limit is not None and count >= self.limit:
                break
