import argparse
from dataclasses import dataclass
from typing import Optional

@dataclass
class InferConfig:
    image_path: str
    output_path: str
    gpu: Optional[str]
    input_size: int
    checkpoint_dir: Optional[str]
    use_gpu_mem_growth: bool

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CBAM Inference (TF1 compatible)")
    p.add_argument("--image_path", type=str, required=True, help="input image")
    p.add_argument("--output_path", type=str, required=True, help="save path (png/jpg)")
    p.add_argument("--checkpoint_dir", type=str, default=None, help="folder that contains .ckpt files")
    p.add_argument("--input_size", type=int, default=256, help="square size to resize the input")
    p.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES (e.g. '0'); omit to use CPU")
    p.add_argument("--mem_growth", action="store_true", default=True, help="allow GPU memory growth")
    return p

def args_to_cfg(ns: argparse.Namespace) -> InferConfig:
    return InferConfig(
        image_path=ns.image_path,
        output_path=ns.output_path,
        gpu=ns.gpu,
        input_size=ns.input_size,
        checkpoint_dir=ns.checkpoint_dir,
        use_gpu_mem_growth=ns.mem_growth,
    )
