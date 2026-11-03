#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <tuple>
#include <vector>
#include <chrono>
#include <cassert>
#include <iostream>
#include <limits>
#include "sdk_stub.hpp"
#define private public
#include "g1_loco_controller.hpp"
#undef private
using namespace robojudo::g1_loco;
#ifndef TEST_DOFS
#define TEST_DOFS 23
#endif
double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
G1LocoConfig cfg(){G1LocoConfig c;c.num_dofs=TEST_DOFS;c.stiffness=std::vector<double>(TEST_DOFS,20);c.damping=std::vector<double>(TEST_DOFS,1);c.fsm_confirm_timeout=.01;c.regular_confirm_timeout=.5;return c;}
std::vector<double> target(){std::vector<double> q(TEST_DOFS,0);q[3]=q[9]=.3;return q;}
void sample(G1LocoController& c,double age=0){RobotState s(TEST_DOFS);auto q=target();s.motor_state.q.assign(q.begin(),q.end());s.received_time=s.tick_time=now()-age;c.robot_state_buffer_.SetData(s);}
bool throws(const std::function<void()>& f){try{f();return false;}catch(...){return true;}}
void no_mode_requests(){assert(fake_sdk::passive_calls==0 && fake_sdk::damp_calls==0 && fake_sdk::passive_mode_calls==0);}
int main(){int cases=0;
 for(int mode=0;mode<5;++mode){fake_sdk::reset();{G1LocoController c(cfg());sample(c);auto q=target();
  if(mode==0)q[3]=12.57;if(mode==1)q[0]=std::numeric_limits<double>::quiet_NaN();if(mode==2)q[0]=std::numeric_limits<double>::infinity();if(mode==3)q.resize(TEST_DOFS-1);if(mode==4)q[0]=1.;
  assert(throws([&]{c.acquire_user_control(q);}));assert(fake_sdk::writes==0 && fake_sdk::user_calls==0);c.close();}no_mode_requests();++cases;
 }
 {fake_sdk::reset();{G1LocoController c(cfg());sample(c,.2);assert(throws([&]{c.acquire_user_control(target());}));assert(fake_sdk::user_calls==0 && fake_sdk::writes==0);}no_mode_requests();++cases;}
 for(bool active:{false,true}){fake_sdk::reset();{G1LocoController c(cfg());sample(c);if(active)c.acquire_user_control(target());int fsm=fake_sdk::fsm;c.close();assert(fake_sdk::fsm==fsm&&!c.is_publish_enabled());}no_mode_requests();++cases;}
 // USER request rejected/thrown while FSM stays 801: stop streaming, Unitree loco keeps control, retry allowed.
 for(int mode=0;mode<2;++mode){fake_sdk::reset();{G1LocoController c(cfg());sample(c);
  if(mode==0)fake_sdk::user_rc=77;if(mode==1)fake_sdk::throw_user=true;
  if(mode==1)assert(throws([&]{c.acquire_user_control(target());}));else assert(c.acquire_user_control(target())!=0);
  assert(!c.is_publish_enabled());assert(!c.motor_command_buffer_.GetData());
  assert(c.get_authority_state()==AuthorityState::INTERNAL);c.close();}no_mode_requests();++cases;
 }
 // FSM unknown after an accepted USER request: keep streaming the first target, never stop in FSM 1000.
 {fake_sdk::reset();{G1LocoController c(cfg());sample(c);fake_sdk::query_rc_after_user=88;
  assert(c.acquire_user_control(target())==kBridgeFsmUnconfirmed);
  assert(c.is_publish_enabled()&&c.motor_command_buffer_.GetData()&&c.get_authority_state()==AuthorityState::USER_ACTIVE);
  auto count=fake_sdk::writes;c.LowCommandWriter();assert(fake_sdk::writes==count+1);
  assert(c.release_to_walkrun()!=0);assert(c.is_publish_enabled()&&c.get_authority_state()==AuthorityState::USER_ACTIVE);
  fake_sdk::query_rc_after_user=0;assert(c.release_to_walkrun()==0);assert(fake_sdk::fsm==801&&!c.is_publish_enabled());
  c.close();}assert(fake_sdk::passive_mode_calls==0&&fake_sdk::damp_calls==0);++cases;}
 // Firmware left FSM 1000 (operator damp): handback requests nothing.
 {fake_sdk::reset();{G1LocoController c(cfg());sample(c);c.acquire_user_control(target());fake_sdk::fsm=1;
  assert(c.release_to_walkrun()==kBridgeUnexpectedFsm);assert(fake_sdk::passive_calls==0&&fake_sdk::fsm==1);
  assert(!c.is_publish_enabled()&&c.get_authority_state()==AuthorityState::INTERNAL);c.close();}no_mode_requests();++cases;}
 // Hold after failed handback: the query-only sync keeps streaming in FSM 1000 and
 // drops the hold once the operator moved the robot (remote damp); it never requests a mode.
 {fake_sdk::reset();{G1LocoController c(cfg());sample(c);c.acquire_user_control(target());
  fake_sdk::release_rc=55;assert(c.release_to_walkrun()==55&&c.is_publish_enabled());auto calls=fake_sdk::passive_calls;
  assert(c.sync_operator_fsm()==1000&&c.is_publish_enabled()&&c.get_authority_state()==AuthorityState::USER_ACTIVE);
  auto count=fake_sdk::writes;c.LowCommandWriter();assert(fake_sdk::writes==count+1);
  fake_sdk::query_rc=66;assert(c.sync_operator_fsm()<0&&c.is_publish_enabled());fake_sdk::query_rc=0;
  fake_sdk::fsm=1;assert(c.sync_operator_fsm()==1&&!c.is_publish_enabled()&&c.get_authority_state()==AuthorityState::INTERNAL);
  assert(fake_sdk::passive_calls==calls&&fake_sdk::start_calls==0);c.close();}
  assert(fake_sdk::passive_mode_calls==0&&fake_sdk::damp_calls==0);++cases;}
 // WALKRUN -> regular loco (FSM 500) only from confirmed WALKRUN with Unitree loco in control.
 {fake_sdk::reset();{G1LocoController c(cfg());sample(c);c.acquire_user_control(target());
  assert(c.switch_walkrun_to_regular()==kBridgeInvalidState&&fake_sdk::start_calls==0&&fake_sdk::fsm==1000);
  assert(c.release_to_walkrun()==0);assert(c.switch_walkrun_to_regular()==0&&fake_sdk::fsm==500&&fake_sdk::start_calls==1&&fake_sdk::last_set_fsm==500);
  assert(c.switch_walkrun_to_regular()==0&&fake_sdk::start_calls==1);
  sample(c);assert(c.acquire_user_control(target())==kBridgeUnexpectedFsm&&!c.is_publish_enabled());
  fake_sdk::fsm=1;assert(c.switch_walkrun_to_regular()==kBridgeUnexpectedFsm&&fake_sdk::start_calls==1);c.close();}
  assert(fake_sdk::passive_mode_calls==0&&fake_sdk::damp_calls==0);++cases;}
 {fake_sdk::reset();{G1LocoController c(cfg());sample(c);c.acquire_user_control(target());auto q=target();
  for(double v:{.9,0.,.9,0.,.9,0.}){q[0]=v;c.step(q);assert(c.is_publish_enabled());}
  q[4]=1.;c.step(q);assert(fake_sdk::targets.back()[4]==1.F);assert(fake_sdk::fsm==1000);
 }no_mode_requests();++cases;}
 {fake_sdk::reset();{G1LocoController c(cfg());sample(c);c.acquire_user_control(target());sample(c,10.);auto count=fake_sdk::writes;c.LowCommandWriter();assert(fake_sdk::writes==count+1&&fake_sdk::fsm==1000);c.step(target());assert(fake_sdk::writes==count+2);}no_mode_requests();++cases;}
 {fake_sdk::reset();{G1LocoController c(cfg());sample(c);c.acquire_user_control(target());auto q=target();q.resize(TEST_DOFS-1);assert(throws([&]{c.step(q);}));}no_mode_requests();++cases;}
 {fake_sdk::reset();{G1LocoController c(cfg());sample(c);c.acquire_user_control(target());c.release_to_walkrun();assert(fake_sdk::fsm==801);assert(fake_sdk::passive_calls==1);sample(c);assert(c.acquire_user_control(target())==0);c.close();}assert(fake_sdk::passive_calls==1&&fake_sdk::damp_calls==0&&fake_sdk::passive_mode_calls==0);++cases;}
 {fake_sdk::reset();auto bad=cfg();bad.stiffness[0]=std::numeric_limits<double>::quiet_NaN();assert(throws([&]{G1LocoController c(bad);}));assert(fake_sdk::writes==0);no_mode_requests();++cases;}
 {fake_sdk::reset();{G1LocoController c(cfg());auto bad=cfg();bad.damping[0]=-1;assert(throws([&]{c.set_gains(bad.stiffness,bad.damping);}));assert(c.damping_[0]==1);}no_mode_requests();++cases;}
 // Release RPC throws while publishing is stopped: never stay in RELEASING without
 // a hold stream; read FSM back and restore the last command (or confirm WALKRUN).
 {fake_sdk::reset();{G1LocoController c(cfg());sample(c);c.acquire_user_control(target());auto q=target();q[4]=.2;c.step(q);
  fake_sdk::throw_release=true;
  assert(c.release_to_walkrun()==kBridgeRpcException);
  assert(c.is_publish_enabled()&&c.get_authority_state()==AuthorityState::USER_ACTIVE&&fake_sdk::fsm==1000);
  auto count=fake_sdk::writes;c.LowCommandWriter();assert(fake_sdk::writes==count+1&&fake_sdk::targets.back()[4]==.2F);
  fake_sdk::throw_query=true;assert(c.release_to_walkrun()!=0);  // RPC and read-back both throw: still holding
  assert(c.is_publish_enabled()&&c.get_authority_state()==AuthorityState::USER_ACTIVE);
  count=fake_sdk::writes;c.LowCommandWriter();assert(fake_sdk::writes==count+1&&fake_sdk::targets.back()[4]==.2F);
  assert(c.sync_operator_fsm()<0&&c.is_publish_enabled());
  fake_sdk::throw_query=false;fake_sdk::throw_release=false;assert(c.release_to_walkrun()==0&&fake_sdk::fsm==801&&!c.is_publish_enabled());
  c.close();}assert(fake_sdk::passive_mode_calls==0&&fake_sdk::damp_calls==0);++cases;}
 // RPC switched the firmware and then threw: read-back confirms WALKRUN; Start() throwing is an error code.
 {fake_sdk::reset();{G1LocoController c(cfg());sample(c);c.acquire_user_control(target());fake_sdk::release_switches_then_throws=true;
  assert(c.release_to_walkrun()==0&&fake_sdk::fsm==801&&!c.is_publish_enabled()&&c.get_authority_state()==AuthorityState::INTERNAL);
  fake_sdk::throw_start=true;assert(c.switch_walkrun_to_regular()==kBridgeRpcException&&fake_sdk::fsm==801&&fake_sdk::start_calls==3);
  fake_sdk::throw_start=false;assert(c.switch_walkrun_to_regular()==0&&fake_sdk::fsm==500);c.close();}
  assert(fake_sdk::passive_mode_calls==0&&fake_sdk::damp_calls==0);++cases;}
 // Regular loco read back as 501 after a delay: success within regular_confirm_timeout,
 // and the FSM trace records 801 -> 501. Already in 501: no new request.
 {fake_sdk::reset();{G1LocoController c(cfg());sample(c);c.acquire_user_control(target());assert(c.release_to_walkrun()==0);
  fake_sdk::start_lands_fsm=501;fake_sdk::start_delay_queries=15;
  assert(c.switch_walkrun_to_regular()==0&&fake_sdk::fsm==501&&fake_sdk::start_calls==1);
  auto tr=c.get_last_fsm_trace();assert(tr.size()>=2&&std::get<1>(tr.front())==0&&std::get<2>(tr.front())==801);
  assert(std::get<1>(tr.back())==1&&std::get<2>(tr.back())==501&&std::get<0>(tr.back())>.2);  // 15 queries x 20 ms
  assert(c.switch_walkrun_to_regular()==0&&fake_sdk::start_calls==1);
  c.close();}assert(fake_sdk::passive_mode_calls==0&&fake_sdk::damp_calls==0);++cases;}
 // Firmware always ignores the request: regular_attempts requests, then timeout; trace shows FSM stayed 801.
 {fake_sdk::reset();auto rc=cfg();rc.regular_confirm_timeout=.1;{G1LocoController c(rc);sample(c);c.acquire_user_control(target());assert(c.release_to_walkrun()==0);
  fake_sdk::start_delay_queries=-1;
  assert(c.switch_walkrun_to_regular()==kBridgeFsmTimeout&&fake_sdk::fsm==801&&fake_sdk::start_calls==3);
  auto tr=c.get_last_fsm_trace();assert(tr.size()==4);
  for(int i=0;i<4;++i)assert(std::get<1>(tr[i])==i&&std::get<2>(tr[i])==801&&std::get<3>(tr[i])==0);
  auto bad=cfg();bad.regular_confirm_timeout=0;assert(throws([&]{G1LocoController x(bad);}));
  bad=cfg();bad.regular_attempts=0;assert(throws([&]{G1LocoController x(bad);}));
  c.close();}assert(fake_sdk::passive_mode_calls==0&&fake_sdk::damp_calls==0);++cases;}
 // Joint -> motor map: 29DoF is 1:1 (waist roll/pitch, wrist pitch/yaw commanded); 23DoF leaves
 // motors 13/14/20/21/27/28 at zero gains.
 {fake_sdk::reset();{G1LocoController c(cfg());sample(c);c.acquire_user_control(target());auto q=target();q[13]=.1;c.step(q);
  if(TEST_DOFS==29){assert(fake_sdk::targets.back()[13]==.1F);}else{assert(fake_sdk::targets.back()[15]==.1F&&fake_sdk::targets.back()[13]==0.F);}
  assert(throws([&]{G1LocoConfig bad=cfg();bad.num_dofs=27;bad.stiffness.resize(27);bad.damping.resize(27);G1LocoController x(bad);}));
  c.close();}no_mode_requests();++cases;}
 // Firmware reporting walk-run as 802: entry, handback confirmation and FSM 500 all accept it.
 {fake_sdk::reset();fake_sdk::fsm=802;fake_sdk::walkrun_fsm=802;{G1LocoController c(cfg());sample(c);
  assert(c.acquire_user_control(target())==0&&fake_sdk::fsm==1000);
  assert(c.release_to_walkrun()==0&&fake_sdk::fsm==802&&!c.is_publish_enabled());
  assert(c.switch_walkrun_to_regular()==0&&fake_sdk::fsm==500);c.close();}
  assert(fake_sdk::passive_mode_calls==0&&fake_sdk::damp_calls==0);++cases;}
 // Field log 2026-11-03: Start() accepted but ignored while WALKRUN settles. Re-request
 // while still WALKRUN; the FSM mode is recorded per attempt.
 {fake_sdk::reset();auto rc=cfg();rc.regular_confirm_timeout=.1;{G1LocoController c(rc);sample(c);c.acquire_user_control(target());assert(c.release_to_walkrun()==0);
  fake_sdk::start_ignored=2;fake_sdk::fsm_mode=3;
  assert(c.switch_walkrun_to_regular()==0&&fake_sdk::fsm==500&&fake_sdk::start_calls==3);
  auto tr=c.get_last_fsm_trace();assert(std::get<1>(tr.back())==3&&std::get<2>(tr.back())==500&&std::get<3>(tr.back())==3);
  c.close();}assert(fake_sdk::passive_mode_calls==0&&fake_sdk::damp_calls==0);++cases;}
 // Operator switches mode with the remote during the wait: stop, never re-request.
 {fake_sdk::reset();auto rc=cfg();rc.regular_confirm_timeout=.1;{G1LocoController c(rc);sample(c);c.acquire_user_control(target());assert(c.release_to_walkrun()==0);
  fake_sdk::start_delay_queries=-1;fake_sdk::pending_fsm=1;fake_sdk::pending_queries=3;
  assert(c.switch_walkrun_to_regular()==kBridgeUnexpectedFsm&&fake_sdk::fsm==1&&fake_sdk::start_calls==1);
  c.close();}assert(fake_sdk::passive_mode_calls==0&&fake_sdk::damp_calls==0);++cases;}
 // Site test 2026-11-03 (unitree-001): WALKRUN ignores 500 but switches on SetFsmId(501).
 {fake_sdk::reset();auto rc=cfg();rc.regular_fsm_id=501;{G1LocoController c(rc);sample(c);c.acquire_user_control(target());assert(c.release_to_walkrun()==0);
  fake_sdk::start_lands_fsm=501;
  assert(c.switch_walkrun_to_regular()==0&&fake_sdk::fsm==501&&fake_sdk::last_set_fsm==501&&fake_sdk::start_calls==1);
  auto bad=cfg();bad.regular_fsm_id=502;assert(throws([&]{G1LocoController x(bad);}));
  bad.regular_fsm_id=801;assert(throws([&]{G1LocoController x(bad);}));
  c.close();}assert(fake_sdk::passive_mode_calls==0&&fake_sdk::damp_calls==0);++cases;}
 // FSM unreadable after the request: no blind re-request.
 {fake_sdk::reset();auto rc=cfg();rc.regular_confirm_timeout=.1;{G1LocoController c(rc);sample(c);c.acquire_user_control(target());assert(c.release_to_walkrun()==0);
  fake_sdk::start_delay_queries=-1;fake_sdk::query_rc_after_start=77;
  assert(c.switch_walkrun_to_regular()==kBridgeFsmQueryFailed&&fake_sdk::start_calls==1);
  auto tr=c.get_last_fsm_trace();assert(std::get<2>(tr.back())==-77);
  c.close();}assert(fake_sdk::passive_mode_calls==0&&fake_sdk::damp_calls==0);++cases;}
 // Site test 2026-11-03 (unitree-001, charged): SetFsmId(801) from regular loco 501 -> 801.
 // Then entry works; already WALKRUN: no request; not from regular loco (damp, lock stand) or
 // with user control: no request.
 for(int from:{500,501}){fake_sdk::reset();fake_sdk::fsm=from;fake_sdk::start_lands_fsm=801;{G1LocoController c(cfg());sample(c);
  assert(c.switch_regular_to_walkrun()==0&&fake_sdk::fsm==801&&fake_sdk::last_set_fsm==801&&fake_sdk::start_calls==1);
  auto tr=c.get_last_fsm_trace();assert(tr.size()==2&&std::get<1>(tr[0])==0&&std::get<2>(tr[0])==from&&std::get<1>(tr[1])==1&&std::get<2>(tr[1])==801);
  assert(c.switch_regular_to_walkrun()==0&&fake_sdk::start_calls==1);
  assert(c.acquire_user_control(target())==0&&fake_sdk::fsm==1000);
  assert(c.switch_regular_to_walkrun()==kBridgeInvalidState&&fake_sdk::start_calls==1);
  assert(c.release_to_walkrun()==0);
  for(int other:{1,4}){fake_sdk::fsm=other;assert(c.switch_regular_to_walkrun()==kBridgeUnexpectedFsm&&fake_sdk::start_calls==1&&fake_sdk::fsm==other);}
  c.close();assert(c.switch_regular_to_walkrun()==kBridgeClosed);}
  assert(fake_sdk::passive_mode_calls==0&&fake_sdk::damp_calls==0);++cases;}
 // Low battery 2026-11-03: SetFsmId(801) from 501 ended in damp (FSM 1). Stop at once, no
 // second request.
 {fake_sdk::reset();fake_sdk::fsm=501;fake_sdk::start_lands_fsm=1;fake_sdk::start_delay_queries=3;{G1LocoController c(cfg());sample(c);
  assert(c.switch_regular_to_walkrun()==kBridgeUnexpectedFsm&&fake_sdk::fsm==1&&fake_sdk::start_calls==1);
  auto tr=c.get_last_fsm_trace();assert(std::get<1>(tr.back())==1&&std::get<2>(tr.back())==1);
  c.close();}assert(fake_sdk::passive_mode_calls==0&&fake_sdk::damp_calls==0);++cases;}
 // Request ignored: re-request only while still regular loco, walkrun_attempts times; FSM
 // unreadable after the request: no blind re-request; config validation.
 {fake_sdk::reset();auto rc=cfg();rc.walkrun_confirm_timeout=.1;fake_sdk::fsm=501;fake_sdk::start_lands_fsm=801;{G1LocoController c(rc);sample(c);
  fake_sdk::start_ignored=2;assert(c.switch_regular_to_walkrun()==0&&fake_sdk::fsm==801&&fake_sdk::start_calls==3);
  fake_sdk::fsm=501;fake_sdk::start_calls=0;fake_sdk::start_delay_queries=-1;
  assert(c.switch_regular_to_walkrun()==kBridgeFsmTimeout&&fake_sdk::fsm==501&&fake_sdk::start_calls==3);
  assert(c.get_last_fsm_trace().size()==4);
  fake_sdk::start_calls=0;fake_sdk::query_rc_after_start=77;
  assert(c.switch_regular_to_walkrun()==kBridgeFsmQueryFailed&&fake_sdk::start_calls==1);
  auto bad=cfg();bad.walkrun_confirm_timeout=0;assert(throws([&]{G1LocoController x(bad);}));
  bad=cfg();bad.walkrun_attempts=0;assert(throws([&]{G1LocoController x(bad);}));
  c.close();}assert(fake_sdk::passive_mode_calls==0&&fake_sdk::damp_calls==0);++cases;}
 std::cout<<cases<<" entry-only controller scenarios passed ("<<TEST_DOFS<<"DoF, fake SDK, no network)\n";
}
