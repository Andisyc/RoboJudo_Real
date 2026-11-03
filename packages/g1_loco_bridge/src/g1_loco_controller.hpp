#pragma once

#include <array>
#include "mimic_safety.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/common/thread/thread.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/g1/loco/g1_loco_client.hpp>

namespace robojudo::g1_loco {

constexpr int32_t kPassiveFsmId = 1;
constexpr int32_t kRegularLocoFsmId = 500;
// Regular loco as read back on some G1 firmware (seen on a 29DoF unit after
// the remote switch: FsmId 501, FsmMode 0).
constexpr int32_t kRegularLocoV2FsmId = 501;
constexpr int32_t kWalkRunFsmId = 801;
// Some G1 firmware (seen on 29DoF units) reports its walk-run mode as 802.
constexpr int32_t kWalkRunV2FsmId = 802;
constexpr bool IsWalkRunFsm(int32_t fsm_id) {
  return fsm_id == kWalkRunFsmId || fsm_id == kWalkRunV2FsmId;
}
constexpr bool IsRegularLocoFsm(int32_t fsm_id) {
  return fsm_id == kRegularLocoFsmId || fsm_id == kRegularLocoV2FsmId;
}
constexpr bool FsmMatches(int32_t observed, int32_t expected) {
  if (IsWalkRunFsm(expected)) return IsWalkRunFsm(observed);
  if (IsRegularLocoFsm(expected)) return IsRegularLocoFsm(observed);
  return observed == expected;
}
constexpr int32_t kUserControlFsmId = 1000;
constexpr int32_t kBridgeInvalidState = -20001;
constexpr int32_t kBridgeNoRobotState = -20002;
constexpr int32_t kBridgeFsmTimeout = -20003;
constexpr int32_t kBridgeFsmQueryFailed = -20004;
constexpr int32_t kBridgeClosed = -20005;
constexpr int32_t kBridgeUnexpectedFsm = -20006;
// USER request may have taken effect; the first target is still streamed.
constexpr int32_t kBridgeFsmUnconfirmed = -20007;
// A LocoClient RPC threw instead of returning an error code.
constexpr int32_t kBridgeRpcException = -20008;

// G1 23DoF subset: indices into the full 29DoF motor array. A 29DoF
// controller (num_dofs == 29) maps joints 1:1 onto the motor array.
constexpr std::array<std::size_t, 23> kG123DofMotorIndices = {
    0,  1,  2,  3,  4,  5,      // left leg
    6,  7,  8,  9,  10, 11,     // right leg
    12,                         // waist_yaw
    15, 16, 17, 18, 19,         // left arm
    22, 23, 24, 25, 26          // right arm
};
constexpr std::size_t kG1FullDofCount = 29;

template <typename T>
class DataBuffer {
 public:
  void SetData(const T& value) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    data_ = std::make_shared<T>(value);
  }

  std::shared_ptr<const T> GetData() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return data_;
  }

  void Clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    data_.reset();
  }

 private:
  mutable std::shared_mutex mutex_;
  std::shared_ptr<T> data_;
};

enum class AuthorityState : uint8_t {
  INTERNAL = 0,
  ACQUIRING = 1,
  USER_ACTIVE = 2,
  RELEASING = 3,
  FAULT = 4,
  CLOSED = 5,
};

struct G1LocoConfig {
  std::string net_if = "eth0";
  double control_dt = 0.02;
  unsigned short num_dofs = 29;

  std::string lowstate_topic = "rt/lowstate";
  std::string user_lowcmd_topic = "rt/user_lowcmd";

  bool enable_odometry = false;
  std::string sport_state_topic = "rt/odommodestate";

  bool enable_torso_imu = false;
  std::string torso_imu_topic = "rt/secondary_imu";

  double sdk_timeout = 5.0;
  double fsm_confirm_timeout = 1.0;
  // WALKRUN -> regular loco: wait regular_confirm_timeout per request and
  // re-request up to regular_attempts times while the FSM is still WALKRUN
  // (the firmware may accept a request and ignore it).
  double regular_confirm_timeout = 2.0;
  int regular_attempts = 3;
  // FSM id requested for regular loco: 500 (LocoClient::Start()) or 501.
  // unitree-001 (29DoF) ignores 500 from WALKRUN but switches on 501.
  int regular_fsm_id = kRegularLocoFsmId;
  // Regular loco -> WALKRUN on Start: SetFsmId(801), same retry rule (only
  // while the FSM is still regular loco). Any other FSM (e.g. damp) ends the
  // switch at once without another request.
  double walkrun_confirm_timeout = 2.0;
  int walkrun_attempts = 3;

  std::vector<double> stiffness;
  std::vector<double> damping;
};

struct MotorCommand {
  explicit MotorCommand(std::size_t num_motors)
      : q_target(num_motors, 0.0F),
        dq_target(num_motors, 0.0F),
        kp(num_motors, 0.0F),
        kd(num_motors, 0.0F),
        tau_ff(num_motors, 0.0F) {}

  std::vector<float> q_target;
  std::vector<float> dq_target;
  std::vector<float> kp;
  std::vector<float> kd;
  std::vector<float> tau_ff;
};

struct MotorState {
  explicit MotorState(std::size_t num_motors)
      : q(num_motors, 0.0F),
        dq(num_motors, 0.0F),
        tau_est(num_motors, 0.0F) {}

  std::vector<float> q;
  std::vector<float> dq;
  std::vector<float> tau_est;
};

struct ImuState {
  std::array<float, 3> rpy{0.0F, 0.0F, 0.0F};
  std::array<float, 3> gyroscope{0.0F, 0.0F, 0.0F};
  std::array<float, 4> quaternion{1.0F, 0.0F, 0.0F, 0.0F};
  std::array<float, 3> accelerometer{0.0F, 0.0F, 0.0F};
};

struct RobotState {
  explicit RobotState(std::size_t num_motors) : motor_state(num_motors) {
    wireless_remote.fill(0);
  }

  uint32_t tick = 0;
  double received_time = 0;
  double tick_time = 0;
  double state_age_s = 1e9;
  MotorState motor_state;
  ImuState imu_state;
  ImuState torso_imu_state;
  std::array<uint8_t, 40> wireless_remote{};
};

struct SportState {
  std::array<float, 3> position{0.0F, 0.0F, 0.0F};
  std::array<float, 3> velocity{0.0F, 0.0F, 0.0F};
};

class G1LocoController {
 public:
  explicit G1LocoController(const G1LocoConfig& config);
  ~G1LocoController();

  G1LocoController(const G1LocoController&) = delete;
  G1LocoController& operator=(const G1LocoController&) = delete;

  bool self_check();
  int32_t speak_safety_warning(const std::string& text);

  RobotState get_robot_state() const;
  SportState get_sport_state() const;

  AuthorityState get_authority_state() const;
  int32_t get_cached_fsm_id() const;
  int32_t get_last_loco_api_result() const;
  bool is_publish_enabled() const;
  uint64_t get_active_publish_count() const;

  int32_t acquire_user_control(
      const std::vector<double>& initial_pd_target);
  int32_t release_to_walkrun();
  int32_t release_to_passive();
  int32_t switch_walkrun_to_regular();
  // Regular loco (500/501) -> WALKRUN (801). Already WALKRUN returns 0; any
  // other FSM returns kBridgeUnexpectedFsm without a request.
  int32_t switch_regular_to_walkrun();
  // (seconds since the switch began, request attempt, fsm id or -|query
  // error|, fsm mode or -1), one entry per change. Attempt 0 is the state
  // before the first request.
  using FsmTraceEntry = std::tuple<double, int32_t, int32_t, int32_t>;
  // FSM read back during the last switch_walkrun_to_regular() or
  // switch_regular_to_walkrun().
  std::vector<FsmTraceEntry> get_last_fsm_trace() const;
  // Query-only: never requests a mode. If the operator moved the firmware
  // out of FSM 1000 (e.g. remote damp), stop the hold stream. Returns the
  // FSM id, or a negative bridge/SDK error.
  int32_t sync_operator_fsm();

  uint64_t step(const std::vector<double>& pd_target);
  void set_gains(const std::vector<double>& stiffness,
                 const std::vector<double>& damping);
  int32_t close();

 private:
  using LowCmd = unitree_hg::msg::dds_::LowCmd_;
  using LowState = unitree_hg::msg::dds_::LowState_;
  using TorsoImuState = unitree_hg::msg::dds_::IMUState_;
  using DdsSportState = unitree_go::msg::dds_::SportModeState_;

  void LowStateHandler(const void* message);
  void SportStateHandler(const void* message);
  void TorsoImuStateHandler(const void* message);
  void LowCommandWriter();
  bool WriteLowCommandOnce();
  void PublishLowCommandLocked(const MotorCommand& command);

  bool PrimeHoldCommandLocked();
  void DisablePublishingLocked();
  void EnablePublishingLocked();
  void ClearCommandLocked();

  int32_t QueryFsmIdLocked(int32_t& fsm_id);
  // Diagnostics only: the FSM mode (sub-state), or -1 if it cannot be read.
  int32_t QueryFsmModeLocked();
  // timeout_s <= 0 uses fsm_confirm_timeout.
  int32_t WaitForFsmLocked(int32_t expected_fsm_id, double timeout_s = 0.0);
  // Internal-to-internal switch from an is_from FSM to an is_to FSM; records
  // last_fsm_trace_.
  int32_t SwitchInternalFsmLocked(bool (*is_from)(int32_t), bool (*is_to)(int32_t),
                                  int32_t request_fsm_id, int attempts,
                                  double confirm_timeout_s);
  int32_t ReleaseToInternalLocked(
      unitree::robot::g1::InternalFsmMode mode,
      int32_t expected_fsm_id);
  int32_t SettleUnconfirmedAcquireLocked(int32_t reason);
  void RestoreUserControlAfterReleaseFailureLocked(
      const std::shared_ptr<const MotorCommand>& last_command);
  void CloseNoThrow() noexcept;
  std::string CheckEntryTargetLocked(const std::vector<double>& q) const;
  std::size_t MotorIndex(std::size_t joint) const {
    return num_dofs_ == kG1FullDofCount ? joint : kG123DofMotorIndices.at(joint);
  }

  G1LocoConfig config_;
  std::vector<double> stiffness_;
  std::vector<double> damping_;
  unsigned short num_dofs_;

  std::atomic<AuthorityState> authority_state_{AuthorityState::FAULT};
  std::atomic<int32_t> cached_fsm_id_{-1};
  std::vector<FsmTraceEntry> last_fsm_trace_;  // guarded by authority_mutex_
  std::atomic<int32_t> last_loco_api_result_{0};
  std::atomic<uint8_t> mode_machine_{0};
  std::atomic<bool> publish_enabled_{false};
  std::atomic<uint64_t> active_publish_count_{0};
  std::atomic<bool> owns_user_control_{false};
  std::atomic<bool> initialized_{false};
  std::atomic<bool> closed_{false};

  mutable std::mutex authority_mutex_;
  std::mutex publish_mutex_;

  DataBuffer<MotorCommand> motor_command_buffer_;
  DataBuffer<RobotState> robot_state_buffer_;
  DataBuffer<SportState> sport_state_buffer_;
  DataBuffer<ImuState> torso_imu_state_buffer_;

  std::shared_ptr<unitree::robot::g1::LocoClient> loco_client_;
  unitree::robot::ChannelPublisherPtr<LowCmd> lowcmd_publisher_;
  unitree::robot::ChannelSubscriberPtr<LowState> lowstate_subscriber_;
  unitree::robot::ChannelSubscriberPtr<TorsoImuState>
      torso_imu_subscriber_;
  unitree::robot::ChannelSubscriberPtr<DdsSportState>
      sport_state_subscriber_;
  unitree::common::ThreadPtr command_writer_ptr_;
};

}  // namespace robojudo::g1_loco
