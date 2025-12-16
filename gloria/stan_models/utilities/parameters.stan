// Copyright (c) 2025 e-dynamics GmbH and affiliates
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// Model parameters shared by all models
real k_std;            // Base trend growth rate
real m_n;              // Normalized trend offset
vector[S] delta_std;   // Trend rate adjustments
vector[K] beta_n;      // Normalized regressor coefficients
