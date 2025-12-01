from dataclasses import dataclass, field
from typing import Optional
import math

@dataclass
class Config:
    # training
    epoch: int = 6
    learning_rate: float = 2e-4
    beta1: float = 0.5
    smoothing: float = 0.9
    lambda_val: float = 1.0
    train_size: float = math.inf
    save_itr: int = 859

    # io / paths
    dataset: str = "wikiart"   # celebA, mnist, lsun, wikiart
    input_fname_pattern: str = "*.jpg"
    log_dir: str = "logs"
    checkpoint_dir: Optional[str] = "checkpoint"
    sample_dir: Optional[str] = "sample"

    # runtime
    train: bool = True
    crop: bool = True
    visualize: bool = True
    wgan: bool = False
    can: bool = True
    replay: bool = False
    use_resize: bool = False

    # image
    batch_size: int = 16
    sample_size: int = 1
    input_height: int = 256
    input_width: Optional[int] = None
    output_height: int = 256
    output_width: Optional[int] = None

    # optional S3
    use_s3: bool = False
    s3_bucket: Optional[str] = None

    # derived fields
    y_dim: int = field(init=False)

    def finalize(self) -> None:
        # fill widths if None
        if self.input_width is None:
            self.input_width = self.input_height
        if self.output_width is None:
            self.output_width = self.output_height

        # dataset-dependent y_dim (kept from original code)
        if self.dataset == "mnist":
            self.y_dim = 10
        elif self.dataset == "wikiart":
            self.y_dim = 27
        else:
            self.y_dim = 0  # unconditional
