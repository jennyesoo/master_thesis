"""
python -m src.inference.app \
  --image_path /path/to/input.jpg \
  --output_path /path/to/output.png \
  --checkpoint_dir /path/to/ckpt_folder \
  --input_size 256 \
  --gpu 0
"""

import logging
from src.infer.cli import build_arg_parser, args_to_cfg
from src.infer.runner import run_inference

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s | %(message)s")
    parser = build_arg_parser()
    ns = parser.parse_args()
    cfg = args_to_cfg(ns)
    run_inference(cfg)

if __name__ == "__main__":
    main()
