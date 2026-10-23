#pragma once

#include <chrono>
#include <cstdint>
#include <optional>

namespace unitree_cpp {

struct ReceiveAges {
    double lowstate_age_s;
    std::optional<double> torso_imu_age_s;
};

// All times share steady_clock; copying a sample must not refresh its receipt.
inline ReceiveAges compute_receive_ages(
    std::chrono::steady_clock::time_point now,
    std::chrono::steady_clock::time_point lowstate_received_at,
    std::chrono::steady_clock::time_point torso_received_at,
    std::uint64_t torso_receive_sequence) noexcept {
    ReceiveAges ages{
        std::chrono::duration<double>(now - lowstate_received_at).count(),
        std::nullopt,
    };
    if (torso_receive_sequence != 0) {
        ages.torso_imu_age_s = std::chrono::duration<double>(now - torso_received_at).count();
    }
    return ages;
}

}  // namespace unitree_cpp
