"""Revision 2 audit: runtime forwarding ONLY, with a fake environment and no SDK.
The historical filename is retained so an upgraded checkout cannot run the old monitor.
This does not certify the recorded commands as safe or reproduce robot dynamics.
"""
from pathlib import Path
import argparse,importlib.util,json,msgpack,numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('offline_mimic_tests',ROOT/'tests/test_mimic_safety_offline.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=argparse.ArgumentParser();p.add_argument('--logs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
for name in ['g1_run_v034','g1_run_v035','g1_run_v036','g1_run_v034_error']:
 path=a.logs/name/'motion.msgpack'
 if not path.exists():continue
 pipeline=m.make();pipeline._enter_mimic(pipeline.env.get_data(),{});pipeline.env.sent.clear();count=0
 def forbidden(*args,**kwargs):raise AssertionError('entry check invoked in running dance')
 for method in ['state','observation','target','heading']:setattr(pipeline.mimic_guard,method,forbidden)
 with path.open('rb') as f:
  for e in msgpack.Unpacker(f,raw=False):
   if not isinstance(e,dict):break
   if e['kind']!='cycle':continue
   q=np.array(e['payload']['final_target']);pipeline.policy.target=q;pipeline.step();count+=1
   assert pipeline.state==m.State.MIMIC_ACTIVE and len(pipeline.env.sent)==count
   np.testing.assert_equal(pipeline.env.sent[-1],q)
 assert pipeline.env.trips==0 and not pipeline.announcements
 rows.append({'run':name,'cycles':count,'unchanged_runtime_forwarding':True,'runtime_monitoring_enabled':False,
              'note':'Fake environment only. No safety, stability, or fall-prevention claim.'})
a.output.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
