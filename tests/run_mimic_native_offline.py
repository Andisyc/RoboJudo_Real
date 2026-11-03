"""Compile production controller against fake SDK; never load real SDK/DDS."""
from pathlib import Path
import subprocess,tempfile,shutil
ROOT=Path(__file__).resolve().parents[1]
headers=['unitree/idl/go2/SportModeState_.hpp','unitree/idl/hg/IMUState_.hpp','unitree/idl/hg/LowCmd_.hpp','unitree/idl/hg/LowState_.hpp','unitree/common/thread/thread.hpp','unitree/robot/channel/channel_factory.hpp','unitree/robot/channel/channel_publisher.hpp','unitree/robot/channel/channel_subscriber.hpp','unitree/robot/g1/loco/g1_loco_client.hpp','unitree/robot/g1/audio/g1_audio_client.hpp']
with tempfile.TemporaryDirectory(prefix='mimic-offline-') as temp:
 out=Path(temp);shutil.copy(ROOT/'tests/mimic_sdk_stub/sdk_stub.hpp',out/'sdk_stub.hpp')
 for h in headers:
  p=out/h;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('#include "sdk_stub.hpp"\n')
 for pkg,dofs in [('g1_loco_bridge_23dof',23),('g1_loco_bridge_23dof_backpack',23),('g1_loco_bridge_29dof',29)]:
  src=ROOT/'packages'/pkg/'src';exe=out/pkg
  subprocess.run(['c++','-std=c++17','-pthread','-O0',f'-DTEST_DOFS={dofs}','-I'+str(out),'-I'+str(src),str(ROOT/'tests/cpp/test_mimic_controller.cpp'),str(src/'g1_loco_controller.cpp'),'-o',str(exe)],check=True)
  print(pkg,flush=True);subprocess.run([str(exe)],check=True)
