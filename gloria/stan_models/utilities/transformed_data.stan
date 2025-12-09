// Copyright (c) 2025 e-dynamics GmbH and affiliates
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.


// Data transformations shared by all models

matrix[T, S] A = get_changepoint_matrix(t, t_change, T, S);

// Find regressor-wise scales
vector[K] reg_scales;
for (j in 1:K) {
  reg_scales[j] = max(X[, j]) - min(X[, j]);
}

// Scaling factor for beta-prior to guarantee that it drops to 1% of its
// maximum value at beta_max = 1/reg_scales for sigma = 3
vector[K] f_beta = inv_sqrt(-2*log(0.01)*reg_scales^2) / 3;

// Calculate prior scales
// Note: Factor 0.072 is chosen such that with tau=3 the double_exponential
// drops to 1% of its maximum value for delta_max = 1
real<lower=0> delta_scale = 0.072*tau;
vector[K] beta_scale = f_beta.*sigmas;