#pragma once

#include <cstdint>

namespace robojudo::g1_loco_v2 {

constexpr int32_t kPassiveFsmId = 1;
constexpr int32_t kWalkRunFsmId = 801;
constexpr int32_t kWalkRunV2FsmId = 802;
constexpr int32_t kUserControlFsmId = 1000;

constexpr bool IsWalkRunFsm(int32_t fsm_id) {
  return fsm_id == kWalkRunFsmId || fsm_id == kWalkRunV2FsmId;
}

constexpr bool MatchesExpectedFsm(int32_t observed, int32_t expected) {
  return IsWalkRunFsm(expected) ? IsWalkRunFsm(observed) : observed == expected;
}

}  // namespace robojudo::g1_loco_v2
