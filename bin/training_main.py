
import logging
from src.infer.cli import build_arg_parser, args_to_config
from src.infer.logging_utils import setup_logging
from src.infer.runner import run

def main() -> None:
    setup_logging()
    parser = build_arg_parser()
    ns = parser.parse_args()
    cfg = args_to_config(ns)

    # 基本參數總覽
    logging.info("===== Config =====")
    for k, v in sorted(vars(ns).items()):
        logging.info("%s: %s", k, v)

    # S3（可選）
    if cfg.use_s3:
        from aws import bucket_exists
        if not cfg.s3_bucket:
            raise ValueError("use_s3=True, but --s3_bucket is empty.")
        if not bucket_exists(cfg.s3_bucket):
            logging.warning('Given S3 bucket "%s" not found; proceeding without S3.', cfg.s3_bucket)

    run(cfg)

if __name__ == "__main__":
    main()

