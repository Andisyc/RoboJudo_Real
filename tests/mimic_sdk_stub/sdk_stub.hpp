#pragma once
// Test-only SDK substitute: no sockets, no DDS, no real threads or RPCs.
#include <array>
#include <memory>
#include <functional>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>
#define UT_CPU_ID_NONE -1
namespace fake_sdk {
inline int fsm=801, writes=0, user_calls=0, passive_calls=0, release_rc=0, query_rc=0, damp_calls=0, user_rc=0, query_rc_after_user=0;
inline int start_calls=0, start_rc=0, passive_mode_calls=0, walkrun_fsm=801;
// Start(): firmware lands in start_lands_fsm after start_delay_queries GetFsmId calls (-1: never).
inline int start_lands_fsm=500, start_delay_queries=0, pending_fsm=-1, pending_queries=0;
// start_ignored: Start() calls accepted (rc 0) but ignored; fsm_mode: GetFsmMode(); query_rc_after_start: GetFsmId rc once Start() was called.
inline int start_ignored=0, fsm_mode=0, query_rc_after_start=0, last_set_fsm=-1;
inline bool throw_release=false, throw_user=false, throw_query=false, throw_start=false, release_switches_then_throws=false;
inline std::vector<std::array<float,29>> targets;
inline void reset(){fsm=801;writes=user_calls=passive_calls=release_rc=query_rc=0;throw_release=throw_user=throw_query=throw_start=release_switches_then_throws=false;damp_calls=user_rc=query_rc_after_user=0;start_calls=start_rc=passive_mode_calls=0;walkrun_fsm=801;start_lands_fsm=500;start_delay_queries=0;pending_fsm=-1;pending_queries=0;start_ignored=0;fsm_mode=0;query_rc_after_start=0;last_set_fsm=-1;targets.clear();}
}
namespace unitree_hg::msg::dds_ {
struct Motor {
 float q_=0,dq_=0,kp_=0,kd_=0,tau_=0;uint8_t mode_=0;
 float& q(){return q_;}float q()const{return q_;}
 float& dq(){return dq_;}float dq()const{return dq_;}
 float& kp(){return kp_;}float& kd(){return kd_;}float& tau(){return tau_;}
 float tau_est()const{return tau_;}uint8_t& mode(){return mode_;}
};
struct IMUState_ {
 std::array<float,4> quat{1,0,0,0}; std::array<float,3> v{0,0,0};
 auto quaternion()const{return quat;}auto gyroscope()const{return v;}auto accelerometer()const{return v;}auto rpy()const{return v;}
};
struct LowCmd_ {
 std::array<Motor,29> motors;uint8_t modepr=0,modemachine=0;uint32_t checksum=0;
 auto& motor_cmd(){return motors;}auto& mode_pr(){return modepr;}auto& mode_machine(){return modemachine;}auto& crc(){return checksum;}
};
struct LowState_ {
 std::array<Motor,29> motors;IMUState_ imu;std::array<uint8_t,40> remote{};uint32_t checksum=0,tick_=1;
 auto& crc(){return checksum;}uint32_t tick()const{return tick_;}
 auto& motor_state(){return motors;}auto& imu_state(){return imu;}auto& wireless_remote(){return remote;}
 uint8_t mode_machine()const{return 0;}
};
}
namespace unitree_go::msg::dds_ {struct SportModeState_ {std::array<float,3> position()const{return {};}std::array<float,3> velocity()const{return {};}};}
namespace unitree::common {
using ThreadPtr=std::shared_ptr<int>;
template<class... Args>ThreadPtr CreateRecurrentThreadEx(Args...){return std::make_shared<int>(0);}
}
namespace unitree::robot {
struct ChannelFactory {static ChannelFactory* Instance(){static ChannelFactory x;return &x;}void Init(int,const std::string&) {}};
template<class T>struct ChannelPublisher {
 explicit ChannelPublisher(const std::string&){}void InitChannel(){}
 void Write(T& data){++fake_sdk::writes;std::array<float,29> q{};for(size_t i=0;i<29;++i)q[i]=data.motor_cmd()[i].q();fake_sdk::targets.push_back(q);}
};
template<class T>using ChannelPublisherPtr=std::shared_ptr<ChannelPublisher<T>>;
template<class T>struct ChannelSubscriber {explicit ChannelSubscriber(const std::string&){}template<class F>void InitChannel(F,int){}};
template<class T>using ChannelSubscriberPtr=std::shared_ptr<ChannelSubscriber<T>>;
namespace g1 {
enum class InternalFsmMode {PASSIVE,WALKRUN,LAST};
struct LocoClient {
 void SetTimeout(float){} void Init(){}
 int GetFsmId(int& id){if(fake_sdk::throw_query)throw std::runtime_error("query RPC exception");if(fake_sdk::pending_fsm>=0&&fake_sdk::pending_queries--<=0){fake_sdk::fsm=fake_sdk::pending_fsm;fake_sdk::pending_fsm=-1;}id=fake_sdk::fsm;if(fake_sdk::start_calls&&fake_sdk::query_rc_after_start)return fake_sdk::query_rc_after_start;return fake_sdk::user_calls && fake_sdk::query_rc_after_user ? fake_sdk::query_rc_after_user : fake_sdk::query_rc;}
 int GetFsmMode(int& mode){mode=fake_sdk::fsm_mode;return 0;}
 int SwitchToUserCtrl(){++fake_sdk::user_calls;if(fake_sdk::throw_user)throw std::runtime_error("acquire RPC exception");if(fake_sdk::user_rc)return fake_sdk::user_rc;fake_sdk::fsm=1000;return 0;}
 int SwitchToInternalCtrl(InternalFsmMode mode){++fake_sdk::passive_calls;if(mode==InternalFsmMode::PASSIVE)++fake_sdk::passive_mode_calls;if(fake_sdk::throw_release)throw std::runtime_error("RPC exception");if(fake_sdk::release_switches_then_throws){fake_sdk::fsm=mode==InternalFsmMode::PASSIVE?1:fake_sdk::walkrun_fsm;throw std::runtime_error("RPC exception after switch");}if(fake_sdk::release_rc)return fake_sdk::release_rc;fake_sdk::fsm=mode==InternalFsmMode::PASSIVE?1:fake_sdk::walkrun_fsm;return 0;}
 int Start(){++fake_sdk::start_calls;if(fake_sdk::throw_start)throw std::runtime_error("start RPC exception");if(fake_sdk::start_rc)return fake_sdk::start_rc;if(fake_sdk::start_ignored>0){--fake_sdk::start_ignored;return 0;}if(fake_sdk::start_delay_queries==0){fake_sdk::fsm=fake_sdk::start_lands_fsm;}else if(fake_sdk::start_delay_queries>0){fake_sdk::pending_fsm=fake_sdk::start_lands_fsm;fake_sdk::pending_queries=fake_sdk::start_delay_queries;}return 0;}
 int SetFsmId(int id){fake_sdk::last_set_fsm=id;return Start();}
 int Damp(){++fake_sdk::damp_calls;fake_sdk::fsm=1;return 0;}
};
struct AudioClient {void SetTimeout(float){}void Init(){}int TtsMaker(const std::string&,int){return 0;}};
}
}
