#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
namespace robojudo::mimic_safety {
constexpr int kRevision = 2026110310;
constexpr double kMaxEntryAge = 0.1;
constexpr double kFirstDelta = 0.8;
inline std::string CheckEntryTarget(const std::vector<double>& q,
                                    const std::vector<double>& measured) {
  if (q.size()!=measured.size() || (q.size()!=23 && q.size()!=29)) return "entry target size";
  for (size_t i=0;i<q.size();++i) {
    if (!std::isfinite(q[i]) || !std::isfinite(measured[i])) return "nonfinite entry target/state";
    if (std::abs(q[i]-measured[i])>kFirstDelta) return "entry target jump at " + std::to_string(i);
  }
  return "";
}
inline bool Fresh(double age) { return std::isfinite(age) && age>=0 && age<=kMaxEntryAge; }
inline bool ValidGains(const std::vector<double>& kp, const std::vector<double>& kd) {
  if(kp.size()!=kd.size() || (kp.size()!=23 && kp.size()!=29)) return false;
  for(size_t i=0;i<kp.size();++i) if(!std::isfinite(kp[i]) || !std::isfinite(kd[i]) || kp[i]<0 || kd[i]<0 || kp[i]>1000 || kd[i]>100) return false;
  return true;
}
}  // namespace robojudo::mimic_safety
