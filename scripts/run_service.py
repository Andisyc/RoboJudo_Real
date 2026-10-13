# Fix OMP perfmance issue on ARM platform (Jetson)
import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import logging
import threading
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager

logger = logging.getLogger("robojudo")

app = FastAPI(title="RoboJuDo Robot Service", version="1.0.0")


class StartRequest(BaseModel):
    config_name: str = "g1_native_loco_mimic"


# Global state
pipeline: Optional[object] = None
control_thread: Optional[threading.Thread] = None
stop_event = threading.Event()
pipeline_lock = threading.Lock()

# Command queue: API puts commands here, control loop reads and executes
# This avoids calling private _handle_commands directly and ensures thread safety
pending_commands: list[str] = []
commands_lock = threading.Lock()


def control_loop():
    """Background control loop, replaces the while loop in run_pipeline.py."""
    global pipeline
    while not stop_event.is_set() and pipeline and not pipeline.should_stop:
        time_start = time.time()

        # Read pending commands from API
        with commands_lock:
            cmds = pending_commands.copy()
            pending_commands.clear()

        try:
            with pipeline_lock:
                # Inject commands into ctrl_data so pipeline handles them naturally
                if cmds:
                    env_data = pipeline.env.get_data()
                    ctrl_data = pipeline.ctrl_manager.get_ctrl_data(env_data)
                    # Append our commands to any existing ones
                    existing = ctrl_data.get("COMMANDS", [])
                    ctrl_data["COMMANDS"] = list(existing) + cmds
                    # Monkey-patch ctrl_manager to return our modified ctrl_data
                    original_get = pipeline.ctrl_manager.get_ctrl_data
                    pipeline.ctrl_manager.get_ctrl_data = lambda _: ctrl_data
                    try:
                        pipeline.step()
                    finally:
                        pipeline.ctrl_manager.get_ctrl_data = original_get
                else:
                    pipeline.step()
        except Exception as e:
            logger.error(f"Control loop error: {e}")
            # Safety: release to passive on error
            try:
                pipeline.env.release_to_passive()
            except Exception as release_error:
                logger.error(f"Failed to release to passive: {release_error}")
            break

        elapsed = time.time() - time_start
        sleep_time = max(0, pipeline.dt - elapsed)
        time.sleep(sleep_time)


def send_command(cmd: str):
    """Queue a command to be executed in the next control loop iteration."""
    with commands_lock:
        pending_commands.append(cmd)


@app.post("/start")
def start_robot(request: StartRequest):
    """Start the robot pipeline with the given config."""
    global pipeline, control_thread

    if pipeline is not None:
        raise HTTPException(status_code=400, detail="Robot already started")

    try:
        config_manager = ConfigManager(config_name=request.config_name)
        cfg = config_manager.get_cfg()
        pipeline_class = getattr(robojudo.pipeline, cfg.pipeline_type)
        pipeline = pipeline_class(cfg=cfg)
        pipeline.prepare()

        stop_event.clear()
        control_thread = threading.Thread(target=control_loop, daemon=True)
        control_thread.start()

        logger.info(f"Robot service started with config: {request.config_name}")
        return {"status": "started", "config": request.config_name}

    except Exception as e:
        logger.error(f"Failed to start robot: {e}")
        pipeline = None
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop")
def stop_robot():
    """Stop the robot and release control safely."""
    global pipeline, control_thread

    if pipeline is None:
        raise HTTPException(status_code=400, detail="Robot not started")

    stop_event.set()
    if control_thread:
        control_thread.join(timeout=5.0)

    try:
        with pipeline_lock:
            pipeline.close()
        pipeline = None
        logger.info("Robot service stopped")
        return {"status": "stopped"}
    except Exception as e:
        logger.error(f"Error during stop: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def get_status():
    """Get current robot and pipeline status."""
    if pipeline is None:
        return {"state": "not_started"}

    try:
        with pipeline_lock:
            env_status = pipeline.env.get_control_status()
            return {
                "state": pipeline.state.name,
                "env": env_status,
                "policy": getattr(pipeline.policy, "name", None),
                "timestep": getattr(pipeline, "timestep", 0),
            }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enter_mimic")
def enter_mimic():
    """Enter mimic mode (equivalent to pressing Start)."""
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Robot not started")
    send_command("[POLICY_MIMIC]")
    return {"status": "enter_mimic_queued"}


@app.post("/return_loco")
def return_loco():
    """Return to native loco mode (equivalent to pressing Select)."""
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Robot not started")
    send_command("[POLICY_LOCO]")
    return {"status": "return_loco_queued"}


@app.post("/switch_policy")
def switch_policy(direction: str = "next"):
    """Switch policy (equivalent to pressing R1 or L1).

    Args:
        direction: "next" for R1, "last" for L1
    """
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Robot not started")
    if direction not in ("next", "last"):
        raise HTTPException(status_code=400, detail="direction must be 'next' or 'last'")
    cmd = "[POLICY_SWITCH],NEXT" if direction == "next" else "[POLICY_SWITCH],LAST"
    send_command(cmd)
    return {"status": f"switch_policy_{direction}_queued"}


@app.post("/shutdown")
def shutdown():
    """Shutdown robot (equivalent to pressing A)."""
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Robot not started")
    send_command("[SHUTDOWN]")
    return {"status": "shutdown_queued"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
