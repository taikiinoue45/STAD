from pathlib import Path


def clean_up() -> None:

    base = Path(".")

    [p.unlink() for p in base.glob("run.log")]
    [p.unlink() for p in base.glob("*.npy")]
