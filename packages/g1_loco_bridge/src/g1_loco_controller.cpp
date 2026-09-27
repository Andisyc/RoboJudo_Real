#include "g1_loco_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace robojudo::g1_loco {
namespace {

uint32_t Crc32Core(uint32_t* ptr, uint32_t len) {
  uint32_t crc = 0xFFFFFFFF;
  constexpr uint32_t polynomial = 0x04c11db7;

  for (uint32_t i = 0; i < len; ++i) {
    uint32_t xbit = 1U << 31U;
    const uint32_t data = ptr[i];
    for (uint32_t bit = 0; bit < 32; ++bit) {
      if ((crc & 0x80000000U) != 0U) {
        crc = (crc << 1U) ^ polynomial;
      } else {
        crc <<= 1U;
      }
      if ((data & xbit) != 0U) {
        crc ^= polynomial;
      }
      xbit >>= 1U;
    }
  }
  return crc;
}

}  // namespace

G1LocoController::G1LocoController(const G1LocoConfig& config)
    : config_(config),
      stiffness_(config.stiffness),
      damping_(config.damping),
      num_dofs_(config.num_dofs) {
  if (num_dofs_ == 0) {
    throw std::invalid_argument("num_dofs must be positive");
  }
  if (stiffness_.size() != num_dofs_ || damping_.size() != num_dofs_) {
    throw std::invalid_argument(
        "stiffness and damping must match num_dofs");
  }
  if (config_.control_dt <= 0.0 || config_.fsm_confirm_timeout <= 0.0 ||
      config_.sdk_timeout <= 0.0) {
    throw std::invalid_argument("controller timeouts and control_dt must be positive");
  }

  unitree::robot::ChannelFactory::Instance()->Init(0, config_.net_if);

  loco_client_ = std::make_shared<unitree::robot::g1::LocoClient>();
  loco_client_->SetTimeout(static_cast<float>(config_.sdk_timeout));
  loco_client_->Init();

  lowcmd_publisher_.reset(
      new unitree::robot::ChannelPublisher<LowCmd>(config_.user_lowcmd_topic));
  lowcmd_publisher_->InitChannel();

  lowstate_subscriber_.reset(
      new unitree::robot::ChannelSubscriber<LowState>(config_.lowstate_topic));
  lowstate_subscriber_->InitChannel(
      std::bind(&G1LocoController::LowStateHandler, this,
                std::placeholders::_1),
      1);

  if (config_.enable_torso_imu) {
    torso_imu_subscriber_.reset(
        new unitree::robot::ChannelSubscriber<TorsoImuState>(
            config_.torso_imu_topic));
    torso_imu_subscriber_->InitChannel(
        std::bind(&G1LocoController::TorsoImuStateHandler, this,
                  std::placeholders::_1),
        1);
  }

  if (config_.enable_odometry) {
    sport_state_subscriber_.reset(
        new unitree::robot::ChannelSubscriber<DdsSportState>(
            config_.sport_state_topic));
    sport_state_subscriber_->InitChannel(
        std::bind(&G1LocoController::SportStateHandler, this,
                  std::placeholders::_1),
        1);
  }

  command_writer_ptr_ = unitree::common::CreateRecurrentThreadEx(
      "g1_loco_writer", UT_CPU_ID_NONE,
      static_cast<uint32_t>(config_.control_dt * 1e6),
      &G1LocoController::LowCommandWriter, this);

  initialized_.store(true);

  std::lock_guard<std::mutex> lock(authority_mutex_);
  int32_t fsm_id = -1;
  const int32_t result = QueryFsmIdLocked(fsm_id);
  if (result == 0 && fsm_id != kUserControlFsmId) {
    authority_state_.store(AuthorityState::INTERNAL);
  } else {
    // FSM 1000 at process startup is not proof that this process owns control.
    authority_state_.store(AuthorityState::FAULT);
  }
}

G1LocoController::~G1LocoController() { CloseNoThrow(); }

bool G1LocoController::self_check() {
  if (!initialized_.load() || closed_.load()) {
    return false;
  }
  if (!robot_state_buffer_.GetData()) {
    return false;
  }
  if (config_.enable_odometry && !sport_state_buffer_.GetData()) {
    return false;
  }
  if (config_.enable_torso_imu && !torso_imu_state_buffer_.GetData()) {
    return false;
  }

  std::lock_guard<std::mutex> lock(authority_mutex_);
  if (authority_state_.load() == AuthorityState::FAULT &&
      !owns_user_control_.load()) {
    int32_t fsm_id = -1;
    if (QueryFsmIdLocked(fsm_id) == 0 &&
        fsm_id != kUserControlFsmId) {
      authority_state_.store(AuthorityState::INTERNAL);
    }
  }
  return authority_state_.load() != AuthorityState::FAULT;
}

RobotState G1LocoController::get_robot_state() const {
  const auto state = robot_state_buffer_.GetData();
  if (!state) {
    throw std::runtime_error("LowState data is not available");
  }
  return *state;
}

SportState G1LocoController::get_sport_state() const {
  const auto state = sport_state_buffer_.GetData();
  if (!state) {
    throw std::runtime_error("SportModeState data is not available");
  }
  return *state;
}

AuthorityState G1LocoController::get_authority_state() const {
  return authority_state_.load();
}

int32_t G1LocoController::get_cached_fsm_id() const {
  return cached_fsm_id_.load();
}

int32_t G1LocoController::get_last_loco_api_result() const {
  return last_loco_api_result_.load();
}

bool G1LocoController::is_publish_enabled() const {
  return publish_enabled_.load();
}

uint64_t G1LocoController::get_active_publish_count() const {
  return active_publish_count_.load();
}

void G1LocoController::LowStateHandler(const void* message) {
  auto& low_state = *const_cast<LowState*>(
      static_cast<const LowState*>(message));
  if (low_state.crc() !=
      Crc32Core(reinterpret_cast<uint32_t*>(&low_state),
                (sizeof(LowState) >> 2U) - 1U)) {
    std::cerr << "[g1_loco_bridge] LowState CRC error" << std::endl;
    return;
  }

  RobotState state(num_dofs_);
  state.tick = low_state.tick();
  for (std::size_t i = 0; i < num_dofs_; ++i) {
    state.motor_state.q.at(i) = low_state.motor_state().at(i).q();
    state.motor_state.dq.at(i) = low_state.motor_state().at(i).dq();
    state.motor_state.tau_est.at(i) =
        low_state.motor_state().at(i).tau_est();
  }

  state.imu_state.quaternion = low_state.imu_state().quaternion();
  state.imu_state.gyroscope = low_state.imu_state().gyroscope();
  state.imu_state.accelerometer = low_state.imu_state().accelerometer();
  state.imu_state.rpy = low_state.imu_state().rpy();
  std::copy_n(low_state.wireless_remote().begin(),
              state.wireless_remote.size(), state.wireless_remote.begin());

  if (config_.enable_torso_imu) {
    const auto torso_imu = torso_imu_state_buffer_.GetData();
    if (torso_imu) {
      state.torso_imu_state = *torso_imu;
    }
  }

  mode_machine_.store(low_state.mode_machine());
  robot_state_buffer_.SetData(state);
}

void G1LocoController::SportStateHandler(const void* message) {
  const auto& dds_state = *static_cast<const DdsSportState*>(message);
  SportState state;
  state.position = dds_state.position();
  state.velocity = dds_state.velocity();
  sport_state_buffer_.SetData(state);
}

void G1LocoController::TorsoImuStateHandler(const void* message) {
  const auto& dds_imu = *static_cast<const TorsoImuState*>(message);
  ImuState state;
  state.rpy = dds_imu.rpy();
  state.gyroscope = dds_imu.gyroscope();
  state.quaternion = dds_imu.quaternion();
  state.accelerometer = dds_imu.accelerometer();
  torso_imu_state_buffer_.SetData(state);
}

void G1LocoController::LowCommandWriter() {
  static_cast<void>(WriteLowCommandOnce());
}

bool G1LocoController::WriteLowCommandOnce() {
  std::lock_guard<std::mutex> publish_lock(publish_mutex_);
  const AuthorityState authority_state = authority_state_.load();
  if (!publish_enabled_.load() ||
      (authority_state != AuthorityState::ACQUIRING &&
       authority_state != AuthorityState::USER_ACTIVE) ||
      closed_.load()) {
    return false;
  }

  const auto command = motor_command_buffer_.GetData();
  if (!command) {
    return false;
  }

  PublishLowCommandLocked(*command);
  if (authority_state == AuthorityState::USER_ACTIVE) {
    active_publish_count_.fetch_add(1);
  }
  return true;
}

void G1LocoController::PublishLowCommandLocked(
    const MotorCommand& command) {
  LowCmd dds_command;
  dds_command.mode_pr() = 0;
  dds_command.mode_machine() = mode_machine_.load();
  for (std::size_t i = 0; i < num_dofs_; ++i) {
    auto& motor = dds_command.motor_cmd().at(i);
    motor.mode() = 1;
    motor.q() = command.q_target.at(i);
    motor.dq() = command.dq_target.at(i);
    motor.kp() = command.kp.at(i);
    motor.kd() = command.kd.at(i);
    motor.tau() = command.tau_ff.at(i);
  }
  dds_command.crc() = Crc32Core(
      reinterpret_cast<uint32_t*>(&dds_command),
      (sizeof(dds_command) >> 2U) - 1U);
  lowcmd_publisher_->Write(dds_command);
}

bool G1LocoController::PrimeHoldCommandLocked() {
  const auto state = robot_state_buffer_.GetData();
  if (!state) {
    return false;
  }

  MotorCommand command(num_dofs_);
  for (std::size_t i = 0; i < num_dofs_; ++i) {
    command.q_target.at(i) = state->motor_state.q.at(i);
    command.kp.at(i) = static_cast<float>(stiffness_.at(i));
    command.kd.at(i) = static_cast<float>(damping_.at(i));
  }
  motor_command_buffer_.SetData(command);
  return true;
}

void G1LocoController::DisablePublishingLocked() {
  std::lock_guard<std::mutex> publish_lock(publish_mutex_);
  publish_enabled_.store(false);
}

void G1LocoController::EnablePublishingLocked() {
  std::lock_guard<std::mutex> publish_lock(publish_mutex_);
  publish_enabled_.store(true);
}

void G1LocoController::ClearCommandLocked() {
  motor_command_buffer_.Clear();
}

int32_t G1LocoController::QueryFsmIdLocked(int32_t& fsm_id) {
  int sdk_fsm_id = -1;
  const int32_t result = loco_client_->GetFsmId(sdk_fsm_id);
  last_loco_api_result_.store(result);
  if (result == 0) {
    fsm_id = static_cast<int32_t>(sdk_fsm_id);
    cached_fsm_id_.store(fsm_id);
  }
  return result;
}

int32_t G1LocoController::WaitForFsmLocked(int32_t expected_fsm_id) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(
                            config_.fsm_confirm_timeout);
  do {
    int32_t fsm_id = -1;
    if (QueryFsmIdLocked(fsm_id) != 0) {
      return kBridgeFsmQueryFailed;
    }
    if (fsm_id == expected_fsm_id) {
      return 0;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  } while (std::chrono::steady_clock::now() < deadline);
  return kBridgeFsmTimeout;
}

int32_t G1LocoController::EnsurePassiveInternalLocked() {
  int32_t fsm_id = -1;
  if (QueryFsmIdLocked(fsm_id) != 0) {
    authority_state_.store(AuthorityState::FAULT);
    return kBridgeFsmQueryFailed;
  }
  if (fsm_id == kPassiveFsmId) {
    authority_state_.store(AuthorityState::INTERNAL);
    return 0;
  }
  if (fsm_id == kUserControlFsmId) {
    authority_state_.store(AuthorityState::FAULT);
    return kBridgeInvalidState;
  }

  const int32_t damp_result = loco_client_->Damp();
  last_loco_api_result_.store(damp_result);
  if (damp_result != 0) {
    return damp_result;
  }
  const int32_t confirm_result = WaitForFsmLocked(kPassiveFsmId);
  if (confirm_result == 0) {
    authority_state_.store(AuthorityState::INTERNAL);
  }
  return confirm_result;
}

int32_t G1LocoController::acquire_user_control(
    const std::vector<double>& initial_pd_target) {
  std::lock_guard<std::mutex> lock(authority_mutex_);
  if (closed_.load()) {
    return kBridgeClosed;
  }
  if (initial_pd_target.size() != num_dofs_) {
    throw std::invalid_argument(
        "initial_pd_target size must match num_dofs");
  }

  const AuthorityState state = authority_state_.load();
  if (state == AuthorityState::USER_ACTIVE) {
    return 0;
  }
  if (state != AuthorityState::INTERNAL) {
    return kBridgeInvalidState;
  }

  int32_t fsm_id = -1;
  if (QueryFsmIdLocked(fsm_id) != 0) {
    authority_state_.store(AuthorityState::FAULT);
    return kBridgeFsmQueryFailed;
  }
  if (fsm_id != kWalkRunFsmId) {
    return kBridgeUnexpectedFsm;
  }

  MotorCommand command(num_dofs_);
  for (std::size_t i = 0; i < num_dofs_; ++i) {
    command.q_target.at(i) =
        static_cast<float>(initial_pd_target.at(i));
    command.kp.at(i) = static_cast<float>(stiffness_.at(i));
    command.kd.at(i) = static_cast<float>(damping_.at(i));
  }
  motor_command_buffer_.SetData(command);
  active_publish_count_.store(0);

  // Stream the first policy target through the 801 -> 1000 handoff so there
  // is no separate measured-position hold phase before mimic control.
  authority_state_.store(AuthorityState::ACQUIRING);
  EnablePublishingLocked();
  if (!WriteLowCommandOnce()) {
    DisablePublishingLocked();
    authority_state_.store(AuthorityState::INTERNAL);
    ClearCommandLocked();
    return kBridgeNoRobotState;
  }

  const int32_t switch_result = loco_client_->SwitchToUserCtrl();
  last_loco_api_result_.store(switch_result);
  if (switch_result != 0) {
    int32_t observed_fsm = -1;
    if (QueryFsmIdLocked(observed_fsm) == 0 &&
        observed_fsm != kUserControlFsmId) {
      DisablePublishingLocked();
      authority_state_.store(AuthorityState::INTERNAL);
      ClearCommandLocked();
    } else if (observed_fsm == kUserControlFsmId) {
      RestoreUserControlAfterReleaseFailureLocked(
          motor_command_buffer_.GetData());
      WriteLowCommandOnce();
    } else {
      authority_state_.store(AuthorityState::FAULT);
    }
    return switch_result;
  }
  // The request was accepted, so cleanup must assume ownership even if the
  // following FSM confirmation cannot be completed.
  owns_user_control_.store(true);

  const int32_t confirm_result = WaitForFsmLocked(kUserControlFsmId);
  if (confirm_result == 0) {
    authority_state_.store(AuthorityState::USER_ACTIVE);
    if (!WriteLowCommandOnce()) {
      authority_state_.store(AuthorityState::FAULT);
      return kBridgeInvalidState;
    }
    return 0;
  }

  const auto hold_command = motor_command_buffer_.GetData();
  const int32_t fallback_result = loco_client_->SwitchToInternalCtrl(
      unitree::robot::g1::InternalFsmMode::PASSIVE);
  last_loco_api_result_.store(fallback_result);
  if (fallback_result == 0 &&
      WaitForFsmLocked(kPassiveFsmId) == 0) {
    DisablePublishingLocked();
    owns_user_control_.store(false);
    authority_state_.store(AuthorityState::INTERNAL);
    ClearCommandLocked();
  } else {
    int32_t observed_fsm = -1;
    if (QueryFsmIdLocked(observed_fsm) == 0 &&
        observed_fsm != kUserControlFsmId) {
      DisablePublishingLocked();
      owns_user_control_.store(false);
      authority_state_.store(AuthorityState::INTERNAL);
      ClearCommandLocked();
    } else {
      RestoreUserControlAfterReleaseFailureLocked(hold_command);
      WriteLowCommandOnce();
    }
  }
  return confirm_result;
}

void G1LocoController::RestoreUserControlAfterReleaseFailureLocked(
    const std::shared_ptr<const MotorCommand>& last_command) {
  if (last_command) {
    motor_command_buffer_.SetData(*last_command);
  } else if (!PrimeHoldCommandLocked()) {
    authority_state_.store(AuthorityState::FAULT);
    return;
  }
  owns_user_control_.store(true);
  authority_state_.store(AuthorityState::USER_ACTIVE);
  EnablePublishingLocked();
}

int32_t G1LocoController::ReleaseToInternalLocked(
    unitree::robot::g1::InternalFsmMode mode,
    int32_t expected_fsm_id) {
  if (closed_.load()) {
    return kBridgeClosed;
  }

  if (authority_state_.load() == AuthorityState::INTERNAL) {
    owns_user_control_.store(false);
    DisablePublishingLocked();
    ClearCommandLocked();
    int32_t fsm_id = -1;
    if (QueryFsmIdLocked(fsm_id) != 0) {
      authority_state_.store(AuthorityState::FAULT);
      return kBridgeFsmQueryFailed;
    }
    return fsm_id == expected_fsm_id ? 0 : kBridgeUnexpectedFsm;
  }
  if (!owns_user_control_.load()) {
    return kBridgeInvalidState;
  }
  if (authority_state_.load() == AuthorityState::ACQUIRING ||
      authority_state_.load() == AuthorityState::RELEASING) {
    return kBridgeInvalidState;
  }

  const auto last_command = motor_command_buffer_.GetData();
  authority_state_.store(AuthorityState::RELEASING);
  DisablePublishingLocked();
  ClearCommandLocked();

  const int32_t switch_result = loco_client_->SwitchToInternalCtrl(mode);
  last_loco_api_result_.store(switch_result);
  int32_t release_result = switch_result;
  if (switch_result == 0) {
    release_result = WaitForFsmLocked(expected_fsm_id);
  }
  if (release_result == 0) {
    owns_user_control_.store(false);
    authority_state_.store(AuthorityState::INTERNAL);
    return 0;
  }

  int32_t fsm_id = -1;
  const int32_t query_result = QueryFsmIdLocked(fsm_id);
  if (query_result == 0 && fsm_id == expected_fsm_id) {
    owns_user_control_.store(false);
    authority_state_.store(AuthorityState::INTERNAL);
    return 0;
  }
  if (query_result == 0 && fsm_id != kUserControlFsmId) {
    owns_user_control_.store(false);
    authority_state_.store(AuthorityState::INTERNAL);
    ClearCommandLocked();
    return kBridgeUnexpectedFsm;
  }

  // Release started from confirmed user ownership. Until an internal FSM is
  // observed, retain that last confirmed authority and keep the hold command.
  if (query_result != 0 || fsm_id == kUserControlFsmId) {
    RestoreUserControlAfterReleaseFailureLocked(last_command);
    static_cast<void>(WriteLowCommandOnce());
  }
  return query_result == 0 ? release_result : kBridgeFsmQueryFailed;
}

int32_t G1LocoController::release_to_walkrun() {
  std::lock_guard<std::mutex> lock(authority_mutex_);
  return ReleaseToInternalLocked(
      unitree::robot::g1::InternalFsmMode::WALKRUN,
      kWalkRunFsmId);
}

int32_t G1LocoController::release_to_passive() {
  std::lock_guard<std::mutex> lock(authority_mutex_);
  return ReleaseToInternalLocked(
      unitree::robot::g1::InternalFsmMode::PASSIVE,
      kPassiveFsmId);
}

uint64_t G1LocoController::step(const std::vector<double>& pd_target) {
  std::lock_guard<std::mutex> lock(authority_mutex_);
  if (closed_.load()) {
    throw std::runtime_error("G1LocoController is closed");
  }
  if (authority_state_.load() != AuthorityState::USER_ACTIVE ||
      !publish_enabled_.load()) {
    throw std::runtime_error(
        "Cannot publish LowCmd without confirmed user control");
  }
  if (pd_target.size() != num_dofs_) {
    throw std::invalid_argument("pd_target size must match num_dofs");
  }

  MotorCommand command(num_dofs_);
  for (std::size_t i = 0; i < num_dofs_; ++i) {
    command.q_target.at(i) = static_cast<float>(pd_target.at(i));
    command.kp.at(i) = static_cast<float>(stiffness_.at(i));
    command.kd.at(i) = static_cast<float>(damping_.at(i));
  }
  motor_command_buffer_.SetData(command);
  if (!WriteLowCommandOnce()) {
    throw std::runtime_error("Failed to publish active user LowCmd");
  }
  return active_publish_count_.load();
}

void G1LocoController::set_gains(
    const std::vector<double>& stiffness,
    const std::vector<double>& damping) {
  std::lock_guard<std::mutex> lock(authority_mutex_);
  if (stiffness.size() != num_dofs_ || damping.size() != num_dofs_) {
    throw std::invalid_argument(
        "stiffness and damping must match num_dofs");
  }
  stiffness_ = stiffness;
  damping_ = damping;
}

int32_t G1LocoController::close() {
  std::lock_guard<std::mutex> lock(authority_mutex_);
  if (closed_.load()) {
    return 0;
  }

  if (owns_user_control_.load()) {
    const int32_t release_result = ReleaseToInternalLocked(
        unitree::robot::g1::InternalFsmMode::PASSIVE,
        kPassiveFsmId);
    if (release_result != 0 || owns_user_control_.load()) {
      return release_result != 0 ? release_result : kBridgeInvalidState;
    }
  } else if (authority_state_.load() == AuthorityState::INTERNAL) {
    const int32_t passive_result = EnsurePassiveInternalLocked();
    if (passive_result != 0) {
      return passive_result;
    }
  }

  DisablePublishingLocked();
  ClearCommandLocked();
  closed_.store(true);
  initialized_.store(false);
  authority_state_.store(AuthorityState::CLOSED);
  return 0;
}

void G1LocoController::CloseNoThrow() noexcept {
  try {
    const int32_t result = close();
    if (result != 0) {
      std::cerr << "[g1_loco_bridge] close could not confirm control release: "
                << result << std::endl;
    }
  } catch (const std::exception& error) {
    authority_state_.store(AuthorityState::FAULT);
    std::cerr << "[g1_loco_bridge] close failed: " << error.what()
              << std::endl;
  } catch (...) {
    authority_state_.store(AuthorityState::FAULT);
    std::cerr << "[g1_loco_bridge] close failed with unknown error"
              << std::endl;
  }
}

}  // namespace robojudo::g1_loco
