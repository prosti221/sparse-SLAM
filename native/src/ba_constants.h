#pragma once

#include <cmath>

namespace slarnative {

// Mirrors slam/utils/constants.py. COMPUTED, never hand-typed: the
// bits must match the Python side's arithmetic exactly (a mistyped
// constant here is invisible while the Huber is inactive and flips
// solves once it activates - measured in C2c).
inline const double kHuberDelta = std::sqrt(5.991);      // sqrt(chi2 95%)
inline const double kGainThreshold = 1e-4;               // BA_TERMINATE_GAIN_THRESHOLD

}  // namespace slarnative
