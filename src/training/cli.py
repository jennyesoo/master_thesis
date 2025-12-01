
import argparse
from src.training.config import Config

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # training
    p.add_argument("--epoch", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--smoothing", type=float, default=0.9)
    p.add_argument("--lambda_val", type=float, default=1.0)
    p.add_argument("--train_size", type=float, default=float("inf"))
    p.add_argument("--save_itr", type=int, default=859)

    # io
    p.add_argument("--dataset", type=str, default="wikiart")
    p.add_argument("--input_fname_pattern", type=str, default="*.jpg")
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoint")
    p.add_argument("--sample_dir", type=str, default="sample")

    # runtime
    p.add_argument("--train", action="store_true", default=True)
    p.add_argument("--no-train", dest="train", action="store_false")
    p.add_argument("--crop", action="store_true", default=True)
    p.add_argument("--visualize", action="store_true", default=True)
    p.add_argument("--wgan", action="store_true", default=False)
    p.add_argument("--can", action="store_true", default=True)
    p.add_argument("--replay", action="store_true", default=False)
    p.add_argument("--use_resize", action="store_true", default=False)

    # image
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--sample_size", type=int, default=1)
    p.add_argument("--input_height", type=int, default=256)
    p.add_argument("--input_width", type=int, default=None)
    p.add_argument("--output_height", type=int, default=256)
    p.add_argument("--output_width", type=int, default=None)

    # s3 (optional)
    p.add_argument("--use_s3", action="store_true", default=False)
    p.add_argument("--s3_bucket", type=str, default=None)
    return p

def args_to_config(ns: argparse.Namespace) -> Config:
    cfg = Config(
        epoch=ns.epoch,
        learning_rate=ns.learning_rate,
        beta1=ns.beta1,
        smoothing=ns.smoothing,
        lambda_val=ns.lambda_val,
        train_size=ns.train_size,
        save_itr=ns.save_itr,
        dataset=ns.dataset,
        input_fname_pattern=ns.input_fname_pattern,
        log_dir=ns.log_dir,
        checkpoint_dir=ns.checkpoint_dir,
        sample_dir=ns.sample_dir,
        train=ns.train,
        crop=ns.crop,
        visualize=ns.visualize,
        wgan=ns.wgan,
        can=ns.can,
        replay=ns.replay,
        use_resize=ns.use_resize,
        batch_size=ns.batch_size,
        sample_size=ns.sample_size,
        input_height=ns.input_height,
        input_width=ns.input_width,
        output_height=ns.output_height,
        output_width=ns.output_width,
        use_s3=ns.use_s3,
        s3_bucket=ns.s3_bucket,
    )
    cfg.finalize()
    return cfg
