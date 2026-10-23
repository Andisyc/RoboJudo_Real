#include "../src/sensor_evidence.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void expect_age(double actual, double expected) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-12) {
        throw std::runtime_error("receive age has wrong sign, units or timestamp");
    }
}
}

int main() {
    using namespace std::chrono;
    using unitree_cpp::compute_receive_ages;
    const auto now = steady_clock::time_point{} + seconds(100);
    const auto low_received = now - milliseconds(125);
    const auto torso_received = now - milliseconds(250);

    const auto ages = compute_receive_ages(now, low_received, torso_received, 7);
    expect_age(ages.lowstate_age_s, 0.125);
    expect_age(ages.torso_imu_age_s.value(), 0.25);

    const auto missing = compute_receive_ages(now, now - milliseconds(1), torso_received, 0);
    expect_age(missing.lowstate_age_s, 0.001);
    if (missing.torso_imu_age_s.has_value()) {
        throw std::runtime_error("an unreceived torso sample must remain unknown");
    }

    const auto later = now + milliseconds(500);
    const auto same_sample = compute_receive_ages(later, low_received, torso_received, 7);
    expect_age(same_sample.lowstate_age_s, 0.625);
    expect_age(same_sample.torso_imu_age_s.value(), 0.75);
    const auto new_low_old_torso = compute_receive_ages(
        later, later - milliseconds(1), torso_received, 7);
    expect_age(new_low_old_torso.lowstate_age_s, 0.001);
    expect_age(new_low_old_torso.torso_imu_age_s.value(), 0.75);

    const auto epoch = steady_clock::time_point{};
    const auto zero_age = compute_receive_ages(epoch, epoch, epoch,
                                              std::numeric_limits<std::uint64_t>::max());
    expect_age(zero_age.lowstate_age_s, 0.0);
    expect_age(zero_age.torso_imu_age_s.value(), 0.0);
    const auto fresh = compute_receive_ages(now, now - milliseconds(1),
                                           now - milliseconds(1), 8);
    expect_age(fresh.lowstate_age_s, 0.001);
    expect_age(fresh.torso_imu_age_s.value(), 0.001);

    // Emit actual production-helper results for the Python consumer smoke.
    std::cout << std::setprecision(17)
              << "{\"fresh\":{\"lowstate_age_s\":" << fresh.lowstate_age_s
              << ",\"torso_imu_age_s\":" << fresh.torso_imu_age_s.value()
              << "},\"missing\":{\"lowstate_age_s\":" << missing.lowstate_age_s
              << ",\"torso_imu_age_s\":null},\"stale\":{\"lowstate_age_s\":"
              << new_low_old_torso.lowstate_age_s << ",\"torso_imu_age_s\":"
              << new_low_old_torso.torso_imu_age_s.value() << "}}\n";
}
