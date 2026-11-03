#pragma once

// SDK-independent transition logic. Vendored identically in each bridge so
// each package remains independently buildable (including source distributions).
#include <cmath>
#include <cstdint>
#include <exception>
#include <string>

namespace robojudo::normal_mode {
constexpr int32_t kNormalFsm = 500;
enum Code : int32_t {
  OK = 0, DISABLED = -20101, INVALID_AUTHORITY = -20102,
  WRONG_SOURCE = -20103, STALE_STATE = -20104, NOT_STATIONARY = -20105,
  QUERY_FAILED = -20106, REQUEST_FAILED = -20107, CONFIRM_TIMEOUT = -20108,
  INTERRUPTED = -20109, CLOSED = -20110,
};
struct Result {
  int32_t code = INVALID_AUTHORITY;
  std::string stage = "precheck";
  std::string reason;
  int32_t before_fsm = -1;
  int32_t observed_fsm = -1;
  int32_t sdk_request_result = 0;
  int32_t sdk_query_result = 0;
  bool request_sent = false;
  bool request_returned = false;
  bool target_confirmed = false;
  double elapsed_s = 0;
};
struct MotionSample {
  bool fresh = false;
  double xy_speed = 0;
  double angular_speed = 0;
  double tilt = 0;
  double max_joint_speed = 0;
};
struct Limits {
  double settle_timeout_s = 2.0;
  double stable_duration_s = 0.3;
  double confirm_timeout_s = 1.0;
  double poll_s = 0.02;
  double max_xy_speed = 0.05;
  double max_angular_speed = 0.15;
  double max_tilt = 0.20;
  double max_joint_speed = 0.50;
  unsigned confirm_samples = 3;
};
inline bool Stationary(const MotionSample& s, const Limits& l) {
  return std::isfinite(s.xy_speed) && std::isfinite(s.angular_speed) &&
         std::isfinite(s.tilt) && std::isfinite(s.max_joint_speed) &&
         s.xy_speed >= 0 && s.xy_speed <= l.max_xy_speed &&
         s.angular_speed >= 0 && s.angular_speed <= l.max_angular_speed &&
         s.tilt >= 0 && s.tilt <= l.max_tilt &&
         s.max_joint_speed >= 0 && s.max_joint_speed <= l.max_joint_speed;
}

// Call only under the controller's authority mutex, with INTERNAL ownership,
// no user LowCmd publisher, and the feature explicitly enabled.
// Query, start, clock and wait are injected for offline fault tests.
template <class Query, class Start, class Sample, class IsWalkRun,
          class Clock, class Wait>
Result Run(Query query, Start start, Sample sample, IsWalkRun is_walkrun,
           Clock now, Wait wait, const Limits& limits) {
  Result r;
  const double begin = now();
  auto finish = [&](int32_t code, const std::string& reason) {
    r.code = code;
    r.reason = reason;
    r.target_confirmed = code == OK;
    r.elapsed_s = now() - begin;
    return r;
  };
  auto read = [&]() {
    int32_t fsm = -1;
    r.observed_fsm = -1;  // Never present the previous successful query as fresh.
    try {
      r.sdk_query_result = query(fsm);
    } catch (const std::exception& e) {
      r.sdk_query_result = QUERY_FAILED;
      r.reason = e.what();
    } catch (...) {
      r.sdk_query_result = QUERY_FAILED;
      r.reason = "unknown query exception";
    }
    if (r.sdk_query_result != 0) return false;
    r.observed_fsm = fsm;
    return true;
  };
  if (!read()) return finish(QUERY_FAILED, "initial FSM query failed: " + r.reason);
  r.before_fsm = r.observed_fsm;
  const bool already_normal = r.before_fsm == kNormalFsm;
  if (!already_normal && !is_walkrun(r.before_fsm))
    return finish(WRONG_SOURCE, "source must be WALKRUN, or already NORMAL (500)");

  if (!already_normal) {
    r.stage = "settle";
    const double deadline = now() + limits.settle_timeout_s;
    double stable_since = -1;
    bool first = true;
    while (now() < deadline) {
      if (!first && !read()) return finish(QUERY_FAILED, "settle FSM query failed");
      first = false;
      if (now() >= deadline) break;
      if (!is_walkrun(r.observed_fsm))
        return finish(INTERRUPTED, "FSM changed during preflight; no request sent");
      const MotionSample state = sample();
      if (!state.fresh) return finish(STALE_STATE, "fresh LowState and odometry required");
      if (Stationary(state, limits)) {
        if (stable_since < 0) stable_since = now();
        if (now() - stable_since >= limits.stable_duration_s) break;
      } else {
        stable_since = -1;
      }
      wait(limits.poll_s);
    }
    if (now() >= deadline)
      return finish(NOT_STATIONARY, "did not remain stationary before settle deadline");
    // The final query and motion sample above immediately precede this request.
    r.stage = "request";
    r.request_sent = true; // A transport exception does not prove nothing was sent.
    try {
      r.sdk_request_result = start(); // Exactly one Start()/SetFsmId(500).
      r.request_returned = true;
    } catch (const std::exception& e) {
      r.sdk_request_result = REQUEST_FAILED;
      r.reason = e.what();
    } catch (...) {
      r.sdk_request_result = REQUEST_FAILED;
      r.reason = "unknown request exception";
    }
  }

  r.stage = "confirm";
  const double deadline = now() + limits.confirm_timeout_s;
  unsigned consecutive = 0;
  bool saw_normal = false;
  while (now() < deadline) {
    if (!read()) return finish(QUERY_FAILED, "confirmation query failed; mode unknown");
    if (now() >= deadline) break; // One in-flight RPC can overrun this deadline.
    if (r.observed_fsm == kNormalFsm) {
      saw_normal = true;
      if (++consecutive >= limits.confirm_samples)
        return finish(OK, already_normal ? "already NORMAL; confirmed without request" :
          (r.sdk_request_result == 0 ? "NORMAL confirmed" :
           "NORMAL confirmed despite SDK request error; inspect SDK result"));
    } else {
      if (already_normal || saw_normal || !is_walkrun(r.observed_fsm))
        return finish(INTERRUPTED, "FSM changed unexpectedly; no retry or rollback");
      consecutive = 0;
    }
    wait(limits.poll_s);
  }
  return finish(r.request_sent && r.sdk_request_result != 0 ? REQUEST_FAILED : CONFIRM_TIMEOUT,
                "NORMAL was not confirmed; no retry, rollback, Damp, or user-control acquire");
}
} // namespace robojudo::normal_mode
