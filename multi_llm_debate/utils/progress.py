from contextlib import contextmanager
from typing import Any, Iterator, Optional

from tqdm import tqdm

class ProgressManager:
    """Manages nested progress bars to avoid conflicts and provide clear progress tracking."""
    
    def __init__(self):
        self.main_progress: Optional[tqdm] = None
        self.sub_progress: Optional[tqdm] = None
        
    @contextmanager
    def main_bar(self, total: int, desc: str, **kwargs: Any) -> Iterator[tqdm]:
        """Creates the main progress bar."""
        try:
            self.main_progress = tqdm(total=total, desc=desc, position=0, **kwargs)
            yield self.main_progress
        finally:
            if self.main_progress:
                self.main_progress.close()
                self.main_progress = None
                
    @contextmanager
    def sub_bar(self, total: int, desc: str, **kwargs: Any) -> Iterator[tqdm]:
        """Creates a sub-progress bar below the main one."""
        try:
            self.sub_progress = tqdm(
                total=total,
                desc=desc,
                position=1,
                leave=False,
                **kwargs
            )
            yield self.sub_progress
        finally:
            if self.sub_progress:
                self.sub_progress.close()
                self.sub_progress = None

# Global instance
progress = ProgressManager()
