from pathlib import Path


def clean_up() -> None:

    base = Path(".")

    [p.unlink() for p in base.glob("run.log")]

    [p.unlink() for p in base.glob("* - test_*_img.jpg")]
    [p.unlink() for p in base.glob("* - test_*_mask.png")]
    [p.unlink() for p in base.glob("* - test_*_heatmap.npy")]

    [p.unlink() for p in base.glob("* - val_img.jpg")]
    [p.unlink() for p in base.glob("* - val_mask.png")]
    [p.unlink() for p in base.glob("* - val_heatmap.npy")]
