"""One process per local role; file lock released on process exit."""
import os
from pathlib import Path

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class LockBusy(RuntimeError):
    pass


class ProcessLock:
    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        try:
            if os.name == "nt":
                if self.handle.seek(0, os.SEEK_END) == 0:
                    self.handle.write(b"\0")
                    self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            self.handle.close()
            raise LockBusy(f"A process is already running for {self.path.name}.") from exc
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            if os.name == "nt":
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
