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
from robojudo.controller.ctrl_cfgs import KeyboardCtrlCfg
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
    parser.add_argument(
        "--left-shoulder-load", action="store_true",
        help="G1 FADA MuJoCo only: attach 3kg at the left shoulder after forward motion starts.",
    )
    parser.add_argument(
        "--left-shoulder-load-delay-s", type=float, default=None,
        help="Simulation seconds after the first forward command before loading (default: 30; 0: immediate).",
    )
    parser.add_argument(
        "--keyboard-hold", action="store_true",
        help="FADA MuJoCo only: local desktop keyboard, release motion keys to zero their axes",
    )
    parser.add_argument("--freeze-lora", action="store_true", help="Start with current LoRA frozen; F freezes and R unfreezes during execution")
    parser.add_argument(
        "--preset-command", nargs=3, type=float, metavar=("VX", "VY", "YAW"),
        help="FADA MuJoCo only: P starts this local keyboard preset (m/s, m/s, rad/s); X cancels",
    )
    parser.add_argument(
        "--key-speed", nargs="+", action="extend", metavar="KEY=SPEED",
        help="FADA MuJoCo only: override local W/S/A/D/Q/E magnitudes independently (m/s or rad/s)",
    )
    
    args = parser.parse_args()
    if args.left_shoulder_load_delay_s is not None and not args.left_shoulder_load:
        parser.error("--left-shoulder-load-delay-s requires --left-shoulder-load")
    overrides = {}
    for entry in args.key_speed or ():
        key, _, speed = entry.partition("=")
        key = key.lower()
        if len(key) != 1 or key not in "wasdqe" or not speed:
            parser.error("--key-speed expects KEY=SPEED with KEY in W/S/A/D/Q/E")
        if key in overrides:
            parser.error(f"--key-speed repeats key {key!r}")
        try:
            overrides[key] = float(speed)
        except ValueError:
            parser.error(f"--key-speed expects a number for {key!r}")
    args.key_speed = overrides
    return args


def main():
    args = parse_args()
    logger.info(f"Using config: {args.config}")

    # get cfg from register (Import-Time Registry)
    config_manager = ConfigManager(config_name=args.config)
    cfg: RlPipelineCfg = config_manager.get_cfg()
    if args.freeze_lora:
        if cfg.policy.policy_type != "FADAPlannerIDMPolicyAdapter":
            raise ValueError("--freeze-lora requires a FADA Planner-IDM policy")
        cfg.policy.online_adaptation.freeze_lora = True
    if args.keyboard_hold or args.preset_command is not None:
        option = "--preset-command" if args.preset_command is not None else "--keyboard-hold"
        if (cfg.env.env_type != "MujocoEnv" or not cfg.env.is_sim
                or cfg.policy.policy_type != "FADAPlannerIDMPolicyAdapter"):
            raise ValueError(f"{option} requires FADA MuJoCo simulation, never a real-robot profile")
        if len(cfg.ctrl) != 1 or not isinstance(cfg.ctrl[0], KeyboardCtrlCfg):
            raise ValueError(f"{option} requires a single local KeyboardCtrl")
        cfg.ctrl = [cfg.ctrl[0].model_copy(update={"backend": "pynput"})]
    policy_overrides = {}
    if args.preset_command is not None:
        policy_overrides["preset_velocity_command"] = args.preset_command
    if args.key_speed:
        if (cfg.env.env_type != "MujocoEnv" or not cfg.env.is_sim
                or cfg.policy.policy_type != "FADAPlannerIDMPolicyAdapter"):
            raise ValueError("--key-speed requires FADA MuJoCo simulation, never a real-robot profile")
        if len(cfg.ctrl) != 1 or not isinstance(cfg.ctrl[0], KeyboardCtrlCfg):
            raise ValueError("--key-speed requires a single local KeyboardCtrl")
        policy_overrides["keyboard_speed_overrides"] = {
            **cfg.policy.keyboard_speed_overrides, **args.key_speed,
        }
    if policy_overrides:
        cfg.policy = type(cfg.policy).model_validate({**cfg.policy.model_dump(), **policy_overrides})
    if args.keyboard_hold:
        cfg.policy.keyboard_command_mode = "hold"
    if args.left_shoulder_load:
        if (cfg.env.env_type != "MujocoEnv" or not cfg.env.is_sim
                or cfg.robot != "g1" or cfg.policy.policy_type != "FADAPlannerIDMPolicyAdapter"):
            raise ValueError("--left-shoulder-load requires G1 FADA MuJoCo simulation")
        overrides: dict[str, bool | float] = {"left_shoulder_load": True}
        if args.left_shoulder_load_delay_s is not None:
            overrides["left_shoulder_load_delay_s"] = args.left_shoulder_load_delay_s
        cfg.env = type(cfg.env).model_validate({**cfg.env.model_dump(), **overrides})

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
