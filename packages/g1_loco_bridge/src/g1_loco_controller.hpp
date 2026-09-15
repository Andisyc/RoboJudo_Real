#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
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
constexpr int32_t kWalkRunFsmId = 801;
constexpr int32_t kUserControlFsmId = 1000;
constexpr int32_t kBridgeInvalidState = -20001;
constexpr int32_t kBridgeNoRobotState = -20002;
constexpr int32_t kBridgeFsmTimeout = -20003;
constexpr int32_t kBridgeFsmQueryFailed = -20004;
constexpr int32_t kBridgeClosed = -20005;
constexpr int32_t kBridgeUnexpectedFsm = -20006;

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

  RobotState get_robot_state() const;
  SportState get_sport_state() const;

  AuthorityState get_authority_state() const;
  int32_t get_cached_fsm_id() const;
  int32_t get_last_loco_api_result() const;
  bool is_publish_enabled() const;
  uint64_t get_active_publish_count() const;

  int32_t acquire_user_control();
  int32_t release_to_walkrun();
  int32_t release_to_passive();

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
  int32_t WaitForFsmLocked(int32_t expected_fsm_id);
  int32_t EnsurePassiveInternalLocked();
  int32_t ReleaseToInternalLocked(
      unitree::robot::g1::InternalFsmMode mode,
      int32_t expected_fsm_id);
  void RestoreUserControlAfterReleaseFailureLocked(
      const std::shared_ptr<const MotorCommand>& last_command);
  void CloseNoThrow() noexcept;

  G1LocoConfig config_;
  std::vector<double> stiffness_;
  std::vector<double> damping_;
  unsigned short num_dofs_;

  std::atomic<AuthorityState> authority_state_{AuthorityState::FAULT};
  std::atomic<int32_t> cached_fsm_id_{-1};
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
