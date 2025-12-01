import logging
from typing import Tuple
from src.training.config import Config
from src.training.paths import resolve_paths
from src.training.utils import show_all_variables, visualize  # keep your original utils
from src.training.model import DCGAN

# TensorFlow 1.x 相容層
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def _build_session() -> tf.compat.v1.Session:
    cfg = tf.compat.v1.ConfigProto()
    cfg.gpu_options.allow_growth = True  # 動態記憶體
    return tf.compat.v1.Session(config=cfg)

def _build_dcgan(sess: tf.compat.v1.Session, cfg: Config, paths) -> DCGAN:
    common_kwargs = dict(
        sess=sess,
        input_width=cfg.input_width,
        input_height=cfg.input_height,
        output_width=cfg.output_width,
        output_height=cfg.output_height,
        batch_size=cfg.batch_size,
        sample_num=cfg.sample_size,
        use_resize=cfg.use_resize,
        replay=cfg.replay,
        y_dim=cfg.y_dim if cfg.y_dim > 0 else None,
        smoothing=cfg.smoothing,
        lamb=cfg.lambda_val,
        dataset_name=cfg.dataset,
        input_fname_pattern=cfg.input_fname_pattern,
        crop=cfg.crop,
        checkpoint_dir=paths.checkpoint_dir,
        sample_dir=paths.sample_dir,
        wgan=cfg.wgan,
        can=cfg.can,
    )
    return DCGAN(**common_kwargs)

def run(cfg: Config) -> None:
    # 準備路徑
    paths = resolve_paths(
        base_log_dir=cfg.log_dir,
        dataset=cfg.dataset,
        is_can=cfg.can,
        lr=cfg.learning_rate,
        imsize=cfg.input_height,
        batch_size=cfg.batch_size,
        checkpoint_dir=cfg.checkpoint_dir,
        sample_dir=cfg.sample_dir,
    )
    logging.info("Log dir: %s", paths.log_dir)
    logging.info("Checkpoint dir: %s", paths.checkpoint_dir)
    logging.info("Sample dir: %s", paths.sample_dir)

    # 建立 TF1 Session 與模型
    with _build_session() as sess:
        dcgan = _build_dcgan(sess, cfg, paths)
        show_all_variables()

        if cfg.train:
            logging.info("Start training...")
            dcgan.train(cfg)       # 你的 DCGAN.train 會讀 cfg.learning_rate 等欄位
        else:
            logging.info("Start testing...")
            dcgan.test(cfg)

        if cfg.visualize:
            logging.info("Visualizing...")
            visualize(sess, dcgan, cfg, OPTION=0)
