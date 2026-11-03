"""Pure ONNX first-frame replay. No SDK. Recorded linear velocity was unavailable: use zero."""
from pathlib import Path
import argparse,importlib.util,json,numpy as np,onnxruntime as ort
from scipy.spatial.transform import Rotation as R
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('safety',ROOT/'robojudo/tools/mimic_safety.py');safety=importlib.util.module_from_spec(spec);spec.loader.exec_module(safety)
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
for fixture in sorted((ROOT/'tests/fixtures/mimic_safety').glob('*.json')):
 d=json.loads(fixture.read_text());path=ROOT/'assets/models/g1/beyondmimic'/(d['model']+'.onnx')
 session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']);meta=session.get_modelmeta().custom_metadata_map
 names=meta['joint_names'].split(',');default=np.fromstring(meta['default_joint_pos'],sep=',');scale=np.fromstring(meta['action_scale'],sep=',');anchor=meta['body_names'].split(',').index(meta['anchor_body_name']);size=session.get_inputs()[0].shape[1]
 def infer(obs):return session.run(None,{'obs':np.asarray(obs,np.float32)[None,:],'time_step':np.array([[0]],np.float32)})
 zero=infer(np.zeros(size));refyaw=R.from_quat(zero[4].squeeze()[anchor][[1,2,3,0]]).as_euler('xyz')[2]
 ref=d['cycle']['inference']['observation_reference'];imu=d['state']['base_imu'];rot=R.from_quat(np.array(imu['quaternion'])[[1,2,3,0]]);current_yaw=rot.as_euler('xyz')[2]
 ids=[d['metadata']['hardware_joint_names'].index(n)for n in names];order=[names.index(n)for n in d['metadata']['environment_joint_names']]
 q=np.array(d['state']['motor']['q'])[ids];dq=np.array(d['state']['motor']['dq'])[ids];rq=R.from_euler('z',refyaw).inv()*R.from_quat(np.array(ref['body_quat_w'])[anchor][[1,2,3,0]])
 row={'run':d['run'],'linear_velocity_assumption':'zero; complete observation not recorded','state_offset_s':d['state_offset_s']}
 for mode,yaw in [('old_origin',np.deg2rad(d['origin_yaw_deg'])),('reanchored',current_yaw)]:
  base=R.from_euler('z',yaw).inv()*rot;ori=(base.inv()*rq).as_matrix()[:,:2].flatten()
  obs=np.r_[ref['joint_pos'],ref['joint_vel'],np.zeros(3),ori,np.zeros(3),imu['gyroscope'],q-default,dq,zero[0].ravel()]
  raw=infer(obs)[0].ravel();target=(raw*scale+default)[order];guard=safety.MimicSafetyGuard(d['metadata']['environment_joint_names']);verdict='accepted'
  try:guard.observation(obs);guard.target(target,q[order])
  except safety.SafetyViolation as exc:verdict=str(exc)
  row[mode]={'raw_action_rmse_vs_recorded':float(np.sqrt(np.mean((raw-np.array(d['cycle']['inference']['raw_actions']).ravel())**2))),'first_target_guard':verdict,'max_abs_target':float(np.max(abs(target))),'left_knee_target':float(target[3])}
 if row['reanchored']['first_target_guard']!='accepted':raise AssertionError(row)
 rows.append(row)
a.output.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
