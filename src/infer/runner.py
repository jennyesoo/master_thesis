# src/inference/runner.py
from __future__ import annotations
import os
import glob
import logging
import tensorflow as tf
from src.infer.cli import InferConfig
from src.infer.io_utils import load_image_rgb, save_image_rgb
from src.infer.model import build_cbam_autoencoder

tf.compat.v1.disable_eager_execution()

def _configure_env(cfg: InferConfig) -> None:
    if cfg.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def _build_session(mem_growth: bool) -> tf.compat.v1.Session:
    conf = tf.compat.v1.ConfigProto()
    if mem_growth:
        conf.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=conf)

def _restore_latest_ckpt(sess: tf.compat.v1.Session, ckpt_dir: str) -> bool:
    saver = tf.compat.v1.train.Saver()
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "*.index")))
    if not cands:
        logging.warning("No checkpoint found under %s; using random weights.", ckpt_dir)
        sess.run(tf.compat.v1.global_variables_initializer())
        return False
    latest = tf.compat.v1.train.latest_checkpoint(ckpt_dir)
    logging.info("Restoring from %s", latest)
    saver.restore(sess, latest)
    return True

def run_inference(cfg: InferConfig) -> None:
    _configure_env(cfg)
    img = load_image_rgb(cfg.image_path, cfg.input_size)

    g = tf.Graph()
    with g.as_default():
        x = tf.compat.v1.placeholder(tf.float32, shape=[None, cfg.input_size, cfg.input_size, 3], name="input")
        y = build_cbam_autoencoder(x, is_training=False)
        saver = tf.compat.v1.train.Saver()

    with _build_session(cfg.use_gpu_mem_growth) as sess:
        if cfg.checkpoint_dir:
            restored = _restore_latest_ckpt(sess, cfg.checkpoint_dir)
            if not restored:
                # fallback: random init for shape sanity
                sess.run(tf.compat.v1.global_variables_initializer())
        else:
            logging.warning("checkpoint_dir not provided; using random weights.")
            sess.run(tf.compat.v1.global_variables_initializer())

        out = sess.run(y, feed_dict={x: img})[0]  # (H,W,3) 0~1
        save_image_rgb(cfg.output_path, out)
        logging.info("Saved inference result to %s", cfg.output_path)
