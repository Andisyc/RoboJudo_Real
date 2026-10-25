<div align="center">
<h1>RoboJuDo 🤖</h1>

*A plug-and-play deploy framework for robots. Just deploy, just do.*

<h3>
🔗 RoboJuDo is part of the FRoM-W1 project, check it out at 👉 
<a href="https://github.com/OpenMOSS/FRoM-W1">
  OpenMOSS / FRoM-W1
</a>
</h3>

<p>
  <!-- Version -->
  <a href="https://github.com/HansZ8/RoboJuDo/releases">
    <img src="https://img.shields.io/github/v/release/HansZ8/RoboJuDo?color=blue&label=version" alt="release"/>
  </a>
  <!-- Platforms -->
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Ubuntu-green" alt="platform"/>
  <!-- Multi-Robot -->
  <img src="https://img.shields.io/badge/robot-UnitreeG1%20%7C%20UnitreeH1%20%7C%20FFTAIgr1-orange" alt="multi-robot"/>
  <!-- Pre Commit -->
  <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white]" alt="pre-commit"/>
  <!-- License -->
  <a href="https://creativecommons.org/licenses/by-nc/4.0/">
    <img src="https://img.shields.io/badge/License-CC--BY--NC--4.0-lightgrey.svg" alt="license"/>
  </a>
</p>
<img src="docs/images/header-demo.gif" width="80%" alt="demo"/>
<br>
<br>
</div>


Tired of projects that release only models but no deployment code? RoboJuDo provides a unified framework that makes policy deployment straightforward and practical.

Our framework highlights:
- **Out-of-the-box**: After setting up RoboJudo, multiple policies can be deployed on both simulation and real robots in minutes: [Quick Start](#quick-start).

- **Decoupled & Modular Design**: With a Python-first design, RoboJuDo makes fast prototyping easy. Environment, Controller, and Policy are modular and freely composable, while minimal code changes allow seamless adaptation across robots and setups: See how we achieve this: [Add a new module](#add-a-new-module).

- **Multi-policy switching**: Seamlessly switch between different policies during a task. Try this: [Multi-Policy Switching](#multi-policy-switch).

- **Light-Weight**: Our framework is lightweight, after 5 minutes of setup, it runs smoothly onboard. By [UnitreeCpp](https://github.com/HansZ8/unitree_cpp), RoboJuDo runs on Unitree G1 without the need for an Ethernet cable.


# 📓Content
 - [📄Introduction](#introduction)
 - [🛠️Easy Setup](#%EF%B8%8Feasy-setup)
 - [📖Quick Start](#quick-start)
 - [🧩 Develop and Contribute](#develop-and-contribute)


# 🗺️Roadmap

<table>
<tr>
<td width="80%">

- [x] [2025.04] Initialized project
- [x] [2025.05] Add support for Unitree G1
- [x] [2025.05] Add support for Unitree H1, FFTAI Gr1T1
- [x] [2025.06] Integrated Unitree C++ SDK
- [x] [2025.08] Add support for beyondmimic
- [x] [2025.09] RoboJuDo Opensource 🎉
- [x] [2025.10] Add support for **ASAP**
  - [x] Implement `deepmimic` and `locomotion`, check [AsapPolicy](./docs/policy.md/#policy--asappolicy)!
  - [x] Preserve original keyboard and joystick mappings
  - [x] Support for **KungfuBot**
- [x] Add policy-switch pipeline with interpolation, check [LocoMimic Example](#loco-mimic-policy-switch-with-interpolation)!
- [x] [2025.11] Add support for **KungfuBot2** ✨, check [KungfuBotGeneralPolicy](./docs/policy.md/#policy--kungfubotgeneralpolicy)!
- [x] [2025.11] Add support for **TWIST** ✨, check [TwistPolicy](./docs/policy.md/#policy--twistpolicy)!
- [ ] Release code for **HugWBC**
- [ ] Release code for **GMT**
- [ ] Upcoming policies...

 

</td>
<td width="20%">

<div align="center">
<img src="docs\images\job.gif" alt="working" width="100%" >
</div>

</td>
</table>

# 📄Introduction

This repository provides a deployment framework for humanoid robots, supporting the use of different policies across different environments (real robots and simulation).  
We decouple the **controller**, **environment**, and **policy**, making it easy for users to add their own policies or environments.  
Experiment configurations can be organized through config files.

The main modules of **RoboJuDo** consist of:

- 🎮 **Controller**: A collection of control signals. It receives external inputs (e.g., joystick, keyboard, motion sequences) and forwards them as `ctrl_data` to the pipeline.  
- 🤖 **Environment**: The execution environment (e.g., Mujoco, real robot). It processes actions provided by the policy and sends real-time sensor data as `env_data` to the pipeline.  
- 🌐 **Policy**: A trained control policy (from various wbc & locomotion works). It generates actions based on information from both the environment and the controller.

Currently, **RoboJuDo** supports the following policy–environment combinations:


<div align="center">
<!-- 
|  | Human2Humanoid | AMO | GMT | HugWBC | BeyondMimic| ... |
|:-------:|:--------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| g1 mujoco | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ... |
| g1 real | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ... |
| h1 mujoco | ✔️ | ❎ | ❎ | ✔️ | ✔️ | ... |
| h1 real | ✔️ | ❎ | ❎ | ✔️ | ❎ |... |
| gr1t1 mujoco | ✔️ | ❎ | ❎ | ❎ | ❎ | ... |
| gr1t1 real | ❎ | ❎ | ❎ | ❎ | ❎ | ... | -->

| Policy | Unitree G1 | Unitree H1 | FFTAI gr1t1 | Ref | Doc | Feature & Note |
|:-------:|:--------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Unitree Official | 🖥️ 🤖 | 🖥️ 🤖 | - | [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) | [UnitreePolicy](./docs/policy.md/#policy--unitreepolicy)|  |
| Unitree Wo Gait | 🖥️ 🤖 | - | - | [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) | [UnitreeWoGaitPolicy](./docs/policy.md/#unitreewogaitpolicy)| no gait |
| Human2Humanoid | 🖥️ 🤖 | 🖥️ 🤖 | 🖥️ | [H2H](https://github.com/LeCAR-Lab/human2humanoid) | [H2HStudentPolicy](./docs/policy.md/#policy--h2hstudentpolicy) | Need PHC submodule |
| Smooth | 🖥️ 🤖 | 🖥️ 🤖 | 🖥️ 🤖⚠️ | [Smooth](https://github.com/zixuan417/smooth-humanoid-locomotion) |  |
| AMO | 🖥️ 🤖 | - | - | [AMO](https://github.com/OpenTeleVision/AMO) | [AmoPolicy](./docs/policy.md/#policy--amopolicy) |  |
| GMT | 🖥️ 🤖 | - | - | [GMT](https://github.com/zixuan417/humanoid-general-motion-tracking) |  |  |
| HugWBC | 🖥️ 🤖 | 🖥️ 🤖 | - | [HugWBC](https://github.com/apexrl/HugWBC) | [HugWbcPolicy](./docs/policy.md/#policy--hugwbcpolicy) |  |
| **BeyondMimic** | 🖥️ 🤖 | - | - | [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) | [BeyondmimicPolicy](./docs/policy.md/#policy--beyondmimicpolicy) | With&Wo SE supported |
| **ASAP** | 🖥️ 🤖 | - | - | [ASAP](https://github.com/LeCAR-Lab/ASAP) | [AsapPolicy](./docs/policy.md/#policy--asappolicy) | deepmimic & locomotion supported |
| KungfuBot<br>**KungfuBot2** | 🖥️ 🤖 | - | - | [PBHC](https://github.com/TeleHuman/PBHC) | [AsapPolicy](./docs/policy.md/#policy--asappolicy)<br>[KungfuBotGeneralPolicy](./docs/policy.md/#policy--kungfubotgeneralpolicy) | Need PHC submodule |
| **TWIST** | 🖥️ 🤖 | - | - | [TWIST](https://github.com/YanjieZe/TWIST) | [TwistPolicy](./docs/policy.md/#policy--twistpolicy) |  |
| ... | ... | ... | ... | ... | ... | ... |
</div>

🖥️ means policy is ready for simulation, while 🤖 means policy has been tested on real robot.


<!-- Refer [Deploy Policy](#amo-policy-for-g1) for usage. -->


# 🛠️Easy Setup

RoboJuDo supports **multiple platforms**, officially tested on **Ubuntu** and **Windows**. 

Robot onboard PCs are also supported.


## 1️⃣ Basic Installation

**Step 1: Clone the repository and create a Python environment**

```bash
git clone https://github.com/HansZ8/RoboJuDo.git
cd RoboJuDo/
# Example using conda
conda create -n robojudo python=3.11 -y
conda activate robojudo
```
**Step 2: Install RoboJuDo**

```bash
# Optional, install cpu version for speed up
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## 2️⃣ Install Optional Modules

RoboJuDo is a **plug-and-play framework**. After a minimal default installation, you can selectively configure and install only the modules you need.

---

**Step 0: \[Optional\] Install Robot SDK**

> *You can skip this for sim2sim and development.*

If you plan to control a real robot, install the corresponding SDK.

For example, see [unitree_setup.md](docs/unitree_setup.md) for Unitree robots.

---

**Step 1: Configure modules**

Edit [submodule_cfg.yaml](./submodule_cfg.yaml) to select modules, by setting `install` as `true`.

> As default, `mujoco_viewer` is selected for sim2sim.

**Step 2: Install modules**

```bash
# Install all required modules
python submodule_install.py

# Or specify modules to install with args
# python submodule_install.py unitree_cpp
```

# 📖Quick Start

`RoboJuDo` is a modular framework where tasks can be flexibly defined by composing configuration files.  
In the following, we use the deployment on G1 as an example.

<!-- 😎For module combinations, we provide ready-to-use config files that can be directly applied.  -->
1. [Run Sim2Sim](#run-robojudo-on-simulation)
2. [Run Sim2Real](#run-robojudo-on-real-robot-🤖)
3. [Deploy More Policies✨](#deploy-more-policies)

## Run RoboJuDo on Simulation

Begin your journey with unitree g1 sim2sim.

> A Xbox controller is needed for control.

```bash
# run the default g1 sim2sim cfg
python scripts/run_pipeline.py
```

To run the migrated UniLab G1 stand-height/walk DAgger student:

```bash
python scripts/run_pipeline.py -c g1_unilab_distill
```

The keyboard controls are `W/S` for forward/backward, `A/D` for lateral
movement, `Q/E` for yaw, and `R/F` for raising/lowering the target height in
0.01 m steps. The target height is clamped to the policy's trained range of
0.650–0.754 m.

You can control the motivation using any Xbox controller:

- `left axes` move forward/backward/lfet/right
- `right axes` turn left/right

<!-- Or a keyboard:

- `wsad` move forward/backward/left/right
- `qe` turn left/right -->

<!-- For cooler policy, run:

```bash
python scripts/run_pipeline.py -c g1_beyondmimic
```
You can control the simulation environment using the Keyboard:

- `shift + <` start the motion play
- `shift + >` pause the motion play
- `shift + |` reset the motion progress
- `~` button: reset robot. -->

## Run RoboJuDo on Real Robot 🤖

### Alert & Disclaimer ⚠️⚠️⚠️
> Before deployment, you'd better first purchase accident insurance to cover any potential incidents that may occur during real-world operation. Policies could cause ⚠️**violent motions**⚠️ when losing balance. Always verify that the emergency stop button (e.g., **A** for default config) works properly.

> Unless you have strong sim-to-real expertise and rigorous safety measures, **DO NOT run these models on real robots**. They are provided for research only, and we disclaim any responsibility for harm, loss, or malfunction.

### Robot Setup

Follow our [setup guide](./docs/unitree_setup.md) to set up the robot sdk on your computer or robot.

### Start RoboJuDo

Open [`g1_cfg.py`](robojudo/config/g1/g1_cfg.py) and modify the `g1_real` config.

Edit the `env_type` and `net_if` according to your robot sdk setup.

```python
class g1_real(g1):
    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeEnv",  # For unitree_sdk2py
        # env_type="UnitreeCppEnv",  # For unitree_cpp, check README for more details
        unitree=G1UnitreeCfg(
            net_if="eth0",  # note: change to your network interface
        ),
    )
```

Refer to [official guide](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_real/README.md#startup-process) to prepare and start the robot.

Then start the pipeline on the real robot:

```bash
python scripts/run_pipeline.py -c g1_real
```

The existing FADA frozen-inference profile keeps its original ONNX default.
After configuring its network interface and deployment prerequisites, use:

```bash
python scripts/run_pipeline.py -c g1_real_fada_planner_idm
```

This is the frozen-inference FADA entry (`online_adaptation.method="none"`).
Set `G1FADAPlannerIDMPolicyCfg.checkpoint_path` to select an explicit checkpoint;
otherwise the existing assets path and filename remain unchanged. There is no
automatic fallback to another file. Paired source PT and inference ONNX files
are supported; separate adapted/LoRA checkpoint formats are not loaded here.

#### Streaming adaptation runtime (migration step 1/6)

The standalone training runtime supports `fada` (legacy streaming FADA) and
`expert_difference` (frozen-base trajectory difference). Both initialize only
IDM Q/V LoRA as trainable; the Planner and original IDM weights stay frozen.
`expert_difference` additionally holds an independent frozen base IDM, copied
before LoRA injection. The initial zero-delta adapters preserve the source
policy's outputs.

For SDK-free module use, with a verified paired PyTorch checkpoint and the
repository's existing Python requirements available (SDK-free is not
dependency-free):

```python
from robojudo.config.g1.policy.g1_fada_policy_cfg import (
    FADAOnlineAdaptationCfg, G1FADAPlannerIDMPolicyCfg,
)
from robojudo.policy.fada.online_runtime import prepare_fada_trainable_runtime

cfg = G1FADAPlannerIDMPolicyCfg(
    checkpoint_path="/absolute/path/to/planner_idm.pt",
    online_adaptation=FADAOnlineAdaptationCfg(method="expert_difference"),
)
runtime = prepare_fada_trainable_runtime(cfg, device="cpu")
```

`runtime.infer_with_snapshot(observation_history, action_history, command)`
returns owned, graph-free histories, nominal future and first action from the
same forward pass. Histories are `(1,H,66)` / `(1,H,29)`; command is `(1,3)`;
the future is `(1,K,66)` and action `(1,29)`. H/K must match the checkpoint.
Learners can call `runtime.policy.idm` with autograd enabled; ordinary inference
remains under `no_grad`. Metadata such as hardware tick, dispatch evidence and
policy version stays outside the 66-D model input.

The runtime, collection, objective and background APIs are connected to the
single-policy `RlPipeline` FADA adapter. Online requests require an explicit
paired PT checkpoint and the native-evidence/actuation conditions below;
unsupported requests fail rather than silently running frozen inference.
FADA models are constructed once before environment initialization and reused
by the wrapper, so missing or incompatible weights also fail before hardware
initialization. Other policies keep their existing initialization order.

Do not infer robot readiness or real-time performance from this module API.
The minimal migrated LoRA/data components retain TTT's Apache-2.0 license in
`robojudo/policy/fada/LICENSE-TTT`; this does not replace the repository license.

#### Causal collection modules (step 2; native verification pending)

`fada.online_collector` provides `FADALegacyTrajectoryCollector` and
`FADAExpertTrajectoryCollector`. Both require the checkpoint architecture and
an explicit positive `max_sensor_age_s`; the expert collector additionally takes
`window_size` (default 10). The age budget must be chosen for the deployment,
not inferred from the offline examples.

Call `observe(projected_state, command, evidence)` before computing/sending the
current action. It closes the previous issued action with this successor
observation. Call `record_action(snapshot, evidence)` only with a confirmed
policy submission, including its monotonic completion time. `FADACollectionResult`
distinguishes acceptance, a completed sample, and a rejection reason.
These interfaces are not yet wired into the robot's control loop.

Evidence carries hardware tick, torso IMU receive sequence/ages, the monotonic
read-start/submission times, episode/control step and policy version, outside
the network input. Missing, repeated, stale, reordered or skipped frames break
incomplete trajectories rather than compressing time. Tick ordering supports
normal uint32 wraparound. Ages must also prove receipt after the prior
submission, using the read-start bracket; no absolute C++/Python clock origins
are subtracted. Local receipt ordering is not proof of hardware actuation or
synchronization between DDS topics.

Legacy windows require H+K-1 closed transitions (35 for H=30/K=6), and emit as
soon as complete; there is no new ten-sample admission rule. Only the future
must keep a constant command, so older commands may remain in the history.
The expert collector retains each actual query until K successor states
complete its pair. `get_window()` reads mature pairs; `acknowledge()` consumes
only explicitly confirmed `(episode_id, control_step)` identities, preserving
later arrivals and in-flight queries. `reset()` clears incomplete data in both
collectors and mature expert pairs. Legacy mature accumulation belongs to
`FADALegacyUpdater`, not a second collector cache.

The native change attaches receive metadata to the same buffered IMU sample;
copying that sample into a LowState snapshot does not refresh its timestamp.
Rebuild `packages/unitree_cpp` against Unitree SDK2, CycloneDDS and pybind11.
The rebuilt `RobotState.sensor_evidence_version` is 1, and `UnitreeCppEnv`
publishes tick, ages, receive sequence and read-start time. Older bindings
produce explicit missing metadata and cannot admit learning samples; frozen
inference remains available. The SDK/DDS build prerequisites were not found
at the local default paths: native build/import and receiver validation remain
pending. SDK-free Python checks do not validate the native extension or actual
DDS traffic. No control/PD protocol was changed and no controller is launched
by these collection modules.

#### Standalone optimization objectives (step 3)

`robojudo.policy.fada.online_updater` consumes an already prepared trainable
runtime. It never reloads the checkpoint, reinjects LoRA or copies another
Planner/base. Create the matching updater once before learning and retain it
across ordinary episode resets.

- `FADALegacyUpdater(runtime)` accepts `FADACausalWindow` via `append_window()`.
  `compute_loss()` uses realized futures, first predicted/issued actions,
  action-coordinate mean MSE and normalized exponential weights
  (`lambda_forget=0.99`). Forward order and train-mode dropout follow the legacy
  donor. `update()` returns `None` for an empty buffer; any non-empty partial
  batch can update, and only success clears the mature buffer. Failures retain
  it but do not promise optimizer rollback.
- `build_expert_samples(runtime.base_idm, mature_pairs, gamma=0.05)` translates
  matched nominal/realized futures through the frozen base and fixes each
  target as the issued snapshot plus the nominal-minus-realized first-action
  difference. `FADAExpertUpdater(runtime, sigma=1e-4)` regresses these fixed
  samples under nominal futures in eval mode with autograd enabled. Its loss
  averages half the squared action norm over samples and adds an L2 pull toward
  the actual deployment-start LoRA parameters, not zero. `compute_loss(samples)`
  returns total/regression/anchor; `update(samples)` returns their scalar stats.
  It neither retains samples nor acknowledges the collector.

Both use LoRA-only Adam (`lr=1e-4`, `weight_decay=0`, `max_grad_norm=1` or `None`
to disable clipping), take one step per update, and restore the IDM's prior
training modes. Their parameters are also exposed by `FADAOnlineAdaptationCfg`.
The official single-policy FADA path uses the coordinator below for scheduling
and exact sample acknowledgement. Do not impose the expert ten-sample gate on
the legacy updater.

The focused SDK-free proof exercises real small Planner-IDM networks, compares
both losses and LoRA gradients to independent donor expressions, checks static
labels/deployment anchoring and unchanged Planner/base weights, and runs both
public update-to-inference paths. It is not native-extension, real-time,
hardware-safety or load-adaptation evidence.

#### Background coordination and complete publication (step 4)

`fada.online_controller.FADAOnlineCoordinator(runtime, online_config)` composes
the existing collectors and objectives. The config method and LoRA settings
must match the prepared runtime, and `max_sensor_age_s` must be explicitly set
to a positive deployment budget. It is intentionally not guessed by this API.
Call `start()` only in a guarded main/lifecycle path, before hardware startup;
do not spawn processes from configuration imports.

A single spawned learner receives an exact, owned initialization snapshot,
including the actual initial adapters and frozen base. It retains its optimizer
and anchor across jobs/resets, never runs a Planner forward, and never shares
mutable model storage with inference. Defaults are CPU/one learner compute
thread, one outstanding job, and startup/update/close budgets of 30/5/1 seconds.
Sample/weight IPC uses owned bytes, not shared torch tensors or on-disk files.
A background transport thread receives complete result messages; control-side
`poll()` reads only an in-process bounded inbox, so even a truncated child reply
does not make it wait for the remaining pipe frame. The monotone stop flag has
no process-owned lock.

Once deployment collection has been triggered, the single control writer uses this order:

1. `observe(state, command, evidence)` closes the preceding physical submission.
2. `poll()` receives validated completions; `apply_pending_at_step_boundary()`
   prepares and applies the entire adapter set before the next forward.
3. `try_submit()` admits at most one eligible batch without waiting for learning.
4. Infer with `runtime.infer_with_snapshot`, perform the real submission, then
   `record_action(snapshot, evidence)` with the applied `policy_version`.

Legacy scheduling requires a non-empty mature batch at a ten-step interval,
not a full expert window. Its ordered transport outbox has capacity 40 by
default; saturation is a terminal error, never silent loss or unbounded replay.
Expert scheduling retains the 40-step warmup, ten-sample window and ten-step
interval. While busy it keeps the collector's latest bounded window; a submitted
batch is held independently until completion. Success confirms only its exact
sample IDs, preserving later arrivals. Learning completion and actor publication
are separate: `status` reports completed/applied versions, in-flight job,
pending counts, update statistics and latency. A result cannot admit another
job until its version is applied.

The collection start is recorded separately from the real episode control-step
index. Warmup and interval eligibility use elapsed control steps from that
collection start, not time spent waiting for a velocity command. The numeric
warmup/window/interval settings and legacy partial-batch behavior are unchanged.

`reset()` requires a new increasing episode identity, clears incomplete data
and unsubmitted expert samples, but retains legacy mature data and accepted
complete jobs. A pre-reset job may finish and its learning is retained; it can
only acknowledge its original IDs and publish after collection is triggered
again at a valid control boundary. No new job/publication is admitted during
preparation or zero-command waiting before that trigger.
Ordinary reset does not reset Adam, the frozen base or deployment anchor.

`request_stop()` invalidates pending publication without waiting for learning.
After actuator safety is handled, call `close()` outside the control loop;
bounded join escalates to termination/kill and checks process/reader exit.
Unexpected exit, timeout, malformed/non-finite weights and outbox saturation
are explicit terminal failures, not retries or inference-only fallbacks.
`wait_for_message()` is an offline/lifecycle helper and must not be called from
the control path. Forced-kill fault injection can emit Python resource-tracker
semaphore-cleanup warnings; verified child/reader exit does not claim that
arbitrary SIGKILL provides graceful cleanup of every IPC resource.

SDK-free smoke exercised both methods through actual spawn/update/publication,
two sequential Adam updates across reset, busy inference, precise late
acknowledgement, malformed and partial replies, learner errors/timeouts,
bounded backlog and shutdown. This does not establish a 20ms device deadline,
DDS correctness, native-extension compatibility or successful load adaptation.
The official Python adapter/pipeline lifecycle is connected as described below.

#### Official control lifecycle (step 5; native acceptance still pending)

`online_adaptation.method` selects `none`, `fada` or `expert_difference`.
The default remains frozen inference with the original PT/ONNX selection.
Online modes require an explicit paired PT, positive `max_sensor_age_s`, the
single-policy `UnitreeCppEnv` G1 position-control path, `act=true`, enabled torso
IMU and rebuilt `sensor_evidence_version=1` bindings. The learner READY handshake
and configuration checks run before the hardware controller is constructed.
Missing capability is an error, not a baseline fallback. Multi-policy online
adaptation is not enabled by this change.

Online queries use `FADAPlaybackController.prepare_query` and one runtime
snapshot forward; they do not commit their predicted action immediately.
After `UnitreeCppEnv.step` returns a body-submission monotonic timestamp,
`confirm_action_submission` commits the raw policy-order action to history and
the causal collector. The outgoing command retains the existing default pose,
execution scale and joint mapping. Absolute PD targets and scaled commands are
never substituted for raw learning labels. Disabled or failed submissions do
not confirm samples.

The timestamp proves only that the local native submission returned. Native
`step` updates the command buffer and invokes its writer, but this is not a DDS
delivery acknowledgement or proof of motor actuation.

Self-check, prepare and dry-run may warm the model but never collect, submit a
new update or publish weights. Their inferred commands are not recorded as
executed history. The first active step starts a new execution history/count;
initial padding is not measured experience. The original 1000-step preparation
and reset at step 900 remain. Ordinary resets retain the learner/Adam/base/anchor
and learned weights while invalidating unconfirmed queries and incomplete
trajectories; active physical history cannot be rewound using snapshot restore.

Online learning has a separate, one-shot velocity gate. After preparation,
zero-command waiting still performs policy inference, actual submissions and
receipt-confirmed raw action-history commits, but does not collect training
samples, submit updates or publish weights. The first effective nonzero
forward/lateral/yaw command during active control triggers collection. The
control history and real control-step index are not reset at this transition;
waiting history remains legitimate model context, not fabricated training
targets. The legacy collector retains its existing mid-episode history rules.

After triggering, `x` clears the velocity command but does not disable learning.
Reset or leaving active control closes the gate and restarts collection-relative
timing, without resetting learned weights/Adam or changing accepted-job retention.
Old in-flight work may complete while waiting, but cannot publish until the
next trigger; warmup governs new submissions, not retained-result publication.
The worker still completes its READY handshake on the cold path and is polled
for failures while waiting. Neither process startup nor a blocking wait is
introduced into control steps. Frozen `none` mode is unchanged.

`RlPipeline.should_stop/request_stop/close` and the entry's `try/finally` cover
normal stop, emergency commands, falls, learner errors, submission errors and
interrupts. Existing actuator damping shutdown is requested before waiting to
reap learning resources. Shutdown disables further policy writes and is
idempotent; partial construction also cleans up resources already obtained.
Cleanup failures remain visible and do not replace the primary exception.
No native gain values or controller protocol were changed, and this does not
claim that every native sending thread has been stopped.

The SDK-free integration exercised real small networks, adapters, wrappers,
pipeline preparation and learner processes with a test-only SDK/sensor/actuator
boundary. It checked all three modes, receipt-only history commits, reversed
joint ordering and non-unit execution scaling, successful update/publication,
reset retention, rejected submissions, startup failure and interrupted main.
No actual SDK2/DDS controller was created. Native build/receiver acceptance,
target-device timing, physical stop delivery and adaptation quality remain
unverified. The comparison entry/configuration delivery is described below.

#### Comparison startup profiles (step 6)

These profiles use the same v024 `close_some_dr` paired checkpoint,
`assets/models/g1/fada/plaidm_v024_close_some_dr/planner_idm_close_some_dr_v001.pt`,
through the shared `_fada_comparison_policy` in `robojudo/config/g1/g1_cfg.py`.
Place this exact checkpoint on each deployment host; the original local artifact
is `FADA_Unofficial/model/plaidm_v024_close_some_dr/planner_idm_close_some_dr_v001.pt`.

| Registered configuration | Method | Learner |
|---|---|---|
| `g1_real_fada_baseline` | `none` | None |
| `g1_real_fada_online` | `fada` | Legacy realized-trajectory regression |
| `g1_real_fada_expert_online` | `expert_difference` | Frozen-base trajectory-difference targets |

All three comparison profiles enable the configured-gait v024 behavior:
1.5Hz gait frequency, initial left/right phases `[0.0, pi]`, and 50Hz control.
Raw actions remain joint-position offsets with unit execution scaling; the
existing default pose, PD gains and preparation/update gates are unchanged.

FADA paired-PT initialization on CPU sets the control process's PyTorch
intra-op thread count to one before loading the model, including online modes.
This bounds the CPU thread pool used by inference; it is process-wide, not a
hard real-time latency guarantee. Learner subprocesses keep their independent
`learner_threads` setting. ONNX and CUDA initialization are unchanged.

For the frozen v024 MuJoCo baseline, run
`python scripts/run_pipeline.py -c g1_fada_baseline` from the repository root in
an environment with a graphical display. This profile disables online updates
and does not initialize Unitree hardware. Do not use `start.sh` for simulation:
it also starts the hardware DDS trajectory recorder.
The profile explicitly reads keyboard input from the launching terminal, even
when `DISPLAY` is set. Focus that terminal: press `w` to move forward at 0.6m/s
or `x` to clear the velocity command; Enter is not required.

FADA self-check/preparation warms the network without committing discarded
predictions. After successful preparation, the policy resets its gait phase,
observation/action histories, last action, keyboard command and IMU yaw reference
before `prepare_done`, matching simulation's cold start. This final policy reset
does not reset the environment, dispatch a motor target or change PD/model
parameters, and is skipped if a stop has been requested.

The original `g1_real_fada_planner_idm` profile still selects
`g1_hang_book_real.onnx`; it is not the shared-PT comparison baseline. The runtime
does not copy, scan for or silently substitute a checkpoint. To use a different
verified paired PT in all three comparison profiles, set `checkpoint_path` in
the shared helper. Matching architecture does not establish weight or behavior
equivalence with the original ONNX or another checkpoint.

**Prerequisites, not an authorization to run hardware:**

- Use an already installed and activated RoboJudo environment with its declared
  dependencies, matching PyTorch and rebuilt `unitree_cpp` evidence-v1 binding.
  The deployment commands below follow the existing Conda/Python entry; they
  do not create an environment, install dependencies or invoke uv.
  Local SDK-free verification separately reused the existing TTT uv environment
  plus cached dependencies; it did not validate a target deployment environment.
- Set `env.unitree.net_if` in the shared parent
  `g1_real_fada_planner_idm` for the deployment host. The comparison profiles
  inherit its G1 position-control/torso-IMU/safety configuration.
  Configure the interface in `cyclonedds.xml` consistently as well; the network
  setup remains a deployment-host responsibility.
- The shared comparison helper sets `max_sensor_age_s=0.02` (20ms, one nominal
  50Hz control period) as an initial **local receive-age** limit. It is not a
  measured device-timing guarantee or a physical actuation-latency budget.
  Missing/non-positive values remain rejected, and the collector still requires
  successor reception after the preceding submission. This setting does not
  open the velocity-triggered learning gate or admit stale/missing evidence.
- Native build/import and actual DDS/IMU reception, device timing and physical
  stop response remain unverified. Do not treat the commands as approval to
  stand, walk or add a load.

From the repository root, choose **one** profile after satisfying those
prerequisites and obtaining authorization for the actual robot run:

As in the existing launch wrapper, select the configured DDS XML:

```bash
export CYCLONEDDS_URI="file://$PWD/cyclonedds.xml"
```

Frozen shared-PT reference:

```bash
python scripts/run_pipeline.py \
  -c g1_real_fada_baseline
```

Legacy streaming FADA:

```bash
python scripts/run_pipeline.py \
  -c g1_real_fada_online
```

Frozen-base trajectory-difference streaming:

```bash
python scripts/run_pipeline.py \
  -c g1_real_fada_expert_online
```

The entry only accepts `-c/--config`; dotted Hydra overrides or a
`--fada-method` argument are not supported. These remain keyboard profiles.
Ctrl-C exits through the common safety-request/cleanup path; the generic
Unitree-controller `A` mapping below is not selected by these profiles.
Independently verify the physical emergency-stop procedure before any live use.
These direct commands do not start the optional trajectory recorder from
`start.sh`. To retain the existing SSH-console/DDS/recorder launch route, use
the wrapper, which now forwards all arguments to the Python entry:

```bash
env -u DISPLAY ./start.sh \
  -c g1_real_fada_expert_online
```

On Linux, unsetting `DISPLAY` selects the `terminal` keyboard backend when the
backend setting is `auto`; a server `DISPLAY=:0` would otherwise select
`pynput`. The wrapper still handles its original DDS configuration and recorder
cleanup. Replace only the configuration name for legacy or frozen comparisons;
no-argument invocation retains the Python entry's existing default.
After preparation, issue a velocity command to start online collection. In the
existing keyboard configuration, `w` requests forward velocity and `x` clears
the requested velocity; neither key is a physical emergency-stop guarantee.

**Reading progress and outputs:** online submitted steps carry
`extras["fada_online"]`: `control_step` is the latest real policy-step index
offered to the collector (`-1` before collection starts).
`collection_start_control_step` and `collection_step` are `None` while waiting
for the first nonzero velocity command, then report its real start index and
elapsed control steps. The worker's existing `state` enum is unchanged.
`learner_version` is the completed learner version; `policy_version` is the
version actually installed for subsequent inference. Completion is not
publication. `pending_samples`, `inflight_samples`, `failure`, `last_update`
and `last_latency_s` expose collection/update state. A ready worker or a zero
policy version while waiting does not mean online learning has started.

These fields are not automatically printed as a terminal progress bar. The
existing `debug.log_obs=True` recorder can write them under each record's
`extras.fada_online` in `logs/log_<timestamp>_<config>/log.msgpack`, with
`config.json` alongside it; it is off by default and can add I/O/latency.
It is debug telemetry, not a guaranteed lossless stop/audit log.
Online weights/Adam state remain in memory: this path does not automatically
export an adapted checkpoint or resume an interrupted learner.

**Evidence and native gap:** the single targeted registered-entry smoke loaded
all three names through the real ConfigManager and `-c` parser, ran real small
networks/optimization through main/prepare/control/finally, kept frozen weights
unchanged, applied learner version 1 in both online modes and reaped their
processes/receiver threads. It substituted only the test hardware boundary and
offline settings (temporary paired PT, known synthetic sensor budget, disabled
debug/kinematics and accelerated scheduling). It was not a production-checkpoint
run, a physical simulation, a 50Hz timing proof or a real robot run.

Step 2 still requires an actual build on a separately authorized host with
Unitree SDK2/CycloneDDS/pybind11 (see `packages/unitree_cpp/CMakeLists.txt` and
the existing SDK setup guide), then loading that newly built extension and
checking version/unknown metadata without constructing `UnitreeController`.
Such an import check alone cannot prove the receiver-age calculations or
real DDS packet freshness. No build, installation or hardware connection was
performed by the step-6 smoke. The earlier forced-kill semaphore-cleanup warning
also remains an explicitly unresolved boundary; graceful smoke completion does
not prove all IPC paths are leak-free.

#### Offline receive-age calculation (step-2 follow-up)

`packages/unitree_cpp/src/sensor_evidence.hpp` now owns the pure C++17
`compute_receive_ages` calculation used by `UnitreeController::get_robot_state`.
It uses the existing steady-clock receipt timestamps, returns seconds, and
leaves an unreceived torso sample unknown. A copied torso sample retains its
original timestamp; a new LowState cannot reset that old torso sample's age.
The helper does not allocate or add an SDK/controller construction path.

`packages/unitree_cpp/tests/test_sensor_evidence.cpp` compiles independently of
SDK/DDS and exercises units, age growth, missing torso data, new LowState with
old torso data and valid zero age. The focused local smoke compiled and ran
this actual production helper with C++17, then passed its emitted ages to both
real Python causal collectors: fresh data was accepted and stale/missing data
rejected. Its temporary executable was removed; no previous suites were rerun.

This is calculation/consumer evidence, not a build of the full SDK extension.
The uv environment used for this check could not find `pybind11`,
`scikit_build_core` or `unitree_cpp`; nothing was installed. The actual
`get_robot_state` SDK path, real pybind field conversion and
`UnitreeCppEnv.get_data` against the newly built extension remain pending.
Constructing a controller is not an offline shortcut: its constructor initializes
DDS, releases the motion-control service and starts command-writer threads,
regardless of the Python environment's `act` flag.


Your robot should move into default pos. 
**During the preparation, put your robot on the ground.**

You can control the real robot using the Unitree controller:
- `A` button: Emergency stop. The robot immediately switches to damping mode. Be careful.
- `left axes` move forward/backward/let/right
- `right axes` turn left/right

## Deploy More Policies

💡Now you’re familiar with RoboJuDo’s config design, it’s time to experience the **amazing variety of policies**!

### BeyondMimic & ASAP

Try the out of box experience of **BeyondMimic** and **ASAP**:

```bash
python scripts/run_pipeline.py -c g1_beyondmimic
python scripts/run_pipeline.py -c g1_asap
```

check documentation [BeyondmimicPolicy](./docs/policy.md/#policy--beyondmimicpolicy) and [AsapPolicy](./docs/policy.md/#policy--asappolicy) for more details.

### Multi-Policy Switch
`g1_switch` config in [g1_cfg.py](robojudo/config/g1/g1_cfg.py) is equipped with Multi-Policy Pipeline.

```bash
python scripts/run_pipeline.py -c g1_switch
```

Xbox Controller:

- `left axes` move forward/backward/left/right
<!-- - `right axes(for/back)` stand higher/squat -->
- `right axes(left/right)` turn left/right

Switch between Unitree Policy and AMO Policy:
- `RB + Dpad[Down]` switch to Unitree Policy
- `RB + Dpad[Up]` switch to AMO Policy

### Loco-Mimic Policy Switch with Interpolation

For deploying **Motion Mimic Policies** with **Locomotion** as backup, we built [LocoMimicPipeline](robojudo/pipeline/rl_loco_mimic_pipeline.py) for multi-policy switching with interpolation, 

Check `g1_locomimic` config in [g1_cfg.py](robojudo/config/g1/g1_cfg.py), and more fancy locomimic configs in [g1_loco_mimic_cfg.py](robojudo/config/g1/g1_loco_mimic_cfg.py).

```bash
python scripts/run_pipeline.py -c g1_locomimic_beyondmimic
python scripts/run_pipeline.py -c g1_locomimic_asap
```

We have the same Keyboard control as ASAP:
- `[` to switch to MotionMimic
- `]` to switch to LocoMotion
- `;` toggle next mimic policy
- `'` toggle prev mimic policy

<div align="center">
<img src="docs/images/locomimic_asap.gif" width="20%" alt="locomimic_asap"/>
</div>

### More Policies

We also provide config files for other policies, check [config_g1](robojudo/config/g1) and [config_h1](robojudo/config/h1) for more details.

In RoboJuDo, we have fully replicated ASAP’s Sim2Real workflow, including all motions. 
Please refer to `g1_locomimic_asap_full` in [g1_loco_mimic_cfg.py](robojudo/config/g1/g1_loco_mimic_cfg.py). This highlights the modular advantages of our framework.


# 🧩Develop and Contribute

## Add a new module

Refer to the documentation on [Policy](docs/policy.md), [Controller](docs/controller.md), [Env](docs/environment.md), create and deploy your own policy in minutes.

(By the way, deploying AMO takes only 30 minutes, and GMT about 1 hour in our framework.)

Or simply create an issue — we will include updates in future releases!

## Contribute to our project

We warmly welcome contributions from the community. Let’s build a strong and open ecosystem for RoboJuDo together!

# 🔗Citation

If you find our work useful, please cite our GitHub repository:

```bibtex
@misc{FRoM-W1,
  author = {Peng Li, Zihan Zhuang, Yangfan Gao, Yi Dong, Sixian Li, Changhao Jiang, Shihan Dou, Zhiheng Xi, Enyu Zhou, Jixuan Huang, Hui Li, Xingjun Ma, Tao Gui, Zuxuan Wu, Qi Zhang, Xuanjing Huang, Yu-Gang Jiang, Xipeng Qiu},
  title = {FRoM-W1: Towards General Humanoid Whole-Body Control with Language Instructions},
  url = {https://github.com/OpenMOSS/FRoM-W1},
  year = {2025}
}

@misc{RoboJuDo,
  author = {Zihan Zhuang, Yi Dong, Peng Li},
  title = {A plug-and-play deploy framework for robots. Just deploy, just do.},
  url = {https://github.com/HansZ8/RoboJuDo},
  year = {2025}
}
```
or star our repo😁

# 🔗 Related Repo

- [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python): used for implementing `UnitreeEnv`.
- [PHC](https://github.com/ZhengyiLuo/PHC): used for implementing the `MotionCtrl` module for OmniH2O.
- [UnitreeCpp](https://github.com/HansZ8/unitree_cpp): our pybind of `unitree_sdk2` used in `UnitreeCppEnv`.
- [ZED Proxy](https://github.com/HansZ8/ZED-Proxy/): ZED Camera Odometry Service.
