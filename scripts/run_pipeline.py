# Fix OMP perfmance issue on ARM platform (Jetson)
import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import logging
import sys
import time

import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline

logger = logging.getLogger("robojudo")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="g1_unilab", # g1_real_loco_mimic
        # RlPipeline: g1 | g1_beyondmimic | g1_asap | g1_real
        # RlLocoMimicPipeline: g1_locomimic_beyondmimic | g1_locomimic_asap | g1_real_loco_mimic
        help="Name of the config class to use",)
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info(f"Using config: {args.config}")

    # get cfg from register (Import-Time Registry)
    config_manager = ConfigManager(config_name=args.config)
    cfg: RlPipelineCfg = config_manager.get_cfg()

    # pipeline_type locate at pipeline_cfgs.py
    pipeline_type = cfg.pipeline_type

    # getattr(obj, name): find attr name in obj, return class (not instance)
    pipeline_class: type[RlPipeline] = getattr(robojudo.pipeline, pipeline_type)
    logger.info(f"Using pipeline: {pipeline_type} -> {pipeline_class}")

    pipeline = pipeline_class(cfg=cfg)
    should_stop = getattr(pipeline, "should_stop", lambda: False)
    request_stop = getattr(pipeline, "request_stop", None)
    close = getattr(pipeline, "close", None)
    try:
        print(f"\ncfg.device: {cfg.device}\n")
        if not cfg.env.is_sim:
            pipeline.prepare()
        while not should_stop():
            time_start = time.time()
            pipeline.step()
            time_diff = time.time() - time_start
            if not cfg.run_fullspeed:
                remaining = pipeline.dt - time_diff
                if remaining > 0:
                    time.sleep(remaining)
                elif not cfg.env.is_sim:
                    logger.error(f"Warning: frame drop -> {remaining}")
                    if remaining < -0.2:
                        logger.critical("Exiting due to excessive frame drop")
                        if request_stop is not None:
                            request_stop()
                        else:
                            pipeline.env.shutdown()
                        break
    except KeyboardInterrupt:
        logger.info("Interrupted; requesting safe shutdown")
    finally:
        has_primary_error = sys.exc_info()[0] is not None
        try:
            if close is not None:
                close()
            else:
                pipeline.env.shutdown()
        except BaseException:
            if not has_primary_error:
                raise
            logger.exception("Shutdown failed while handling the primary exception")


if __name__ == "__main__":
    main()
