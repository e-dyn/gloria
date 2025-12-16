// Copyright (c) 2025 e-dynamics GmbH and affiliates
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

generated quantities {
  vector[K] beta = beta_n ./ x_std;      // Regressor coefficients
  real m = m_n - dot_product(x_mean, beta);  // Trend offset
  
  real k = k_std / t_scale;
  vector[S] delta = delta_std / t_scale;
}