#include <cstring>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "g1_loco_controller.hpp"

namespace py = pybind11;
using robojudo::g1_loco::AuthorityState;
using robojudo::g1_loco::G1LocoConfig;
using robojudo::g1_loco::G1LocoController;
using robojudo::g1_loco::ImuState;
using robojudo::g1_loco::MotorState;
using robojudo::g1_loco::RobotState;
using robojudo::g1_loco::SportState;

namespace {

template <typename T>
T GetOr(const py::dict& config, const char* key, T default_value) {
  return config.contains(key) ? config[key].cast<T>() : default_value;
}

G1LocoConfig ConfigFromDict(const py::dict& config) {
  G1LocoConfig result;
  result.net_if = GetOr<std::string>(config, "net_if", result.net_if);
  result.control_dt = GetOr<double>(config, "control_dt", result.control_dt);
  result.num_dofs = config["num_dofs"].cast<unsigned short>();
  result.lowstate_topic =
      GetOr<std::string>(config, "lowstate_topic", result.lowstate_topic);
  result.user_lowcmd_topic = GetOr<std::string>(
      config, "user_lowcmd_topic", result.user_lowcmd_topic);
  result.enable_odometry =
      GetOr<bool>(config, "enable_odometry", result.enable_odometry);
  result.sport_state_topic = GetOr<std::string>(
      config, "sport_state_topic", result.sport_state_topic);
  result.enable_torso_imu =
      GetOr<bool>(config, "enable_torso_imu", result.enable_torso_imu);
  result.torso_imu_topic = GetOr<std::string>(
      config, "torso_imu_topic", result.torso_imu_topic);
  result.sdk_timeout =
      GetOr<double>(config, "sdk_timeout", result.sdk_timeout);
  result.fsm_confirm_timeout = GetOr<double>(
      config, "fsm_confirm_timeout", result.fsm_confirm_timeout);
  result.stiffness = config["stiffness"].cast<std::vector<double>>();
  result.damping = config["damping"].cast<std::vector<double>>();
  return result;
}

}  // namespace

PYBIND11_MODULE(_g1_loco_bridge, module) {
  module.doc() =
      "RoboJuDo-owned Unitree G1 internal/user control handoff bridge";

  py::enum_<AuthorityState>(module, "AuthorityState")
      .value("INTERNAL", AuthorityState::INTERNAL)
      .value("ACQUIRING", AuthorityState::ACQUIRING)
      .value("USER_ACTIVE", AuthorityState::USER_ACTIVE)
      .value("RELEASING", AuthorityState::RELEASING)
      .value("FAULT", AuthorityState::FAULT)
      .value("CLOSED", AuthorityState::CLOSED)
      .export_values();

  py::class_<G1LocoConfig>(module, "G1LocoConfig")
      .def(py::init<>())
      .def_readwrite("net_if", &G1LocoConfig::net_if)
      .def_readwrite("control_dt", &G1LocoConfig::control_dt)
      .def_readwrite("num_dofs", &G1LocoConfig::num_dofs)
      .def_readwrite("lowstate_topic", &G1LocoConfig::lowstate_topic)
      .def_readwrite("user_lowcmd_topic",
                     &G1LocoConfig::user_lowcmd_topic)
      .def_readwrite("enable_odometry", &G1LocoConfig::enable_odometry)
      .def_readwrite("sport_state_topic",
                     &G1LocoConfig::sport_state_topic)
      .def_readwrite("enable_torso_imu",
                     &G1LocoConfig::enable_torso_imu)
      .def_readwrite("torso_imu_topic", &G1LocoConfig::torso_imu_topic)
      .def_readwrite("sdk_timeout", &G1LocoConfig::sdk_timeout)
      .def_readwrite("fsm_confirm_timeout",
                     &G1LocoConfig::fsm_confirm_timeout)
      .def_readwrite("stiffness", &G1LocoConfig::stiffness)
      .def_readwrite("damping", &G1LocoConfig::damping);

  py::class_<MotorState>(module, "MotorState")
      .def(py::init<std::size_t>())
      .def_readonly("q", &MotorState::q)
      .def_readonly("dq", &MotorState::dq)
      .def_readonly("tau_est", &MotorState::tau_est);

  py::class_<ImuState>(module, "ImuState")
      .def(py::init<>())
      .def_readonly("rpy", &ImuState::rpy)
      .def_readonly("gyroscope", &ImuState::gyroscope)
      .def_readonly("quaternion", &ImuState::quaternion)
      .def_readonly("accelerometer", &ImuState::accelerometer);

  py::class_<RobotState>(module, "RobotState")
      .def(py::init<std::size_t>())
      .def_readonly("tick", &RobotState::tick)
      .def_readonly("motor_state", &RobotState::motor_state)
      .def_readonly("imu_state", &RobotState::imu_state)
      .def_readonly("torso_imu_state", &RobotState::torso_imu_state)
      .def_property_readonly("wireless_remote", [](const RobotState& state) {
        return py::bytes(
            reinterpret_cast<const char*>(state.wireless_remote.data()),
            state.wireless_remote.size());
      });

  py::class_<SportState>(module, "SportState")
      .def(py::init<>())
      .def_readonly("position", &SportState::position)
      .def_readonly("velocity", &SportState::velocity);

  py::class_<G1LocoController>(module, "G1LocoController")
      .def(py::init<const G1LocoConfig&>(), py::arg("config"))
      .def(py::init([](const py::dict& config) {
        return std::make_unique<G1LocoController>(ConfigFromDict(config));
      }))
      .def("self_check", &G1LocoController::self_check)
      .def("get_robot_state", &G1LocoController::get_robot_state)
      .def("get_sport_state", &G1LocoController::get_sport_state)
      .def("get_authority_state",
           &G1LocoController::get_authority_state)
      .def("get_cached_fsm_id", &G1LocoController::get_cached_fsm_id)
      .def("get_last_loco_api_result",
           &G1LocoController::get_last_loco_api_result)
      .def("is_publish_enabled",
           &G1LocoController::is_publish_enabled)
      .def("get_active_publish_count",
           &G1LocoController::get_active_publish_count)
      .def("acquire_user_control",
           &G1LocoController::acquire_user_control,
           py::call_guard<py::gil_scoped_release>())
      .def("release_to_walkrun", &G1LocoController::release_to_walkrun,
           py::call_guard<py::gil_scoped_release>())
      .def("release_to_passive", &G1LocoController::release_to_passive,
           py::call_guard<py::gil_scoped_release>())
      .def("step", &G1LocoController::step, py::arg("pd_target"))
      .def("set_gains", &G1LocoController::set_gains,
           py::arg("stiffness"), py::arg("damping"))
      .def("close", &G1LocoController::close,
           py::call_guard<py::gil_scoped_release>());

  module.attr("USER_CONTROL_FSM_ID") =
      robojudo::g1_loco::kUserControlFsmId;
  module.attr("PASSIVE_FSM_ID") = robojudo::g1_loco::kPassiveFsmId;
  module.attr("WALKRUN_FSM_ID") = robojudo::g1_loco::kWalkRunFsmId;
  module.attr("BRIDGE_INVALID_STATE") =
      robojudo::g1_loco::kBridgeInvalidState;
  module.attr("BRIDGE_NO_ROBOT_STATE") =
      robojudo::g1_loco::kBridgeNoRobotState;
  module.attr("BRIDGE_FSM_TIMEOUT") =
      robojudo::g1_loco::kBridgeFsmTimeout;
  module.attr("BRIDGE_FSM_QUERY_FAILED") =
      robojudo::g1_loco::kBridgeFsmQueryFailed;
  module.attr("BRIDGE_CLOSED") = robojudo::g1_loco::kBridgeClosed;
  module.attr("BRIDGE_UNEXPECTED_FSM") =
      robojudo::g1_loco::kBridgeUnexpectedFsm;
}
