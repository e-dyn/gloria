// Copyright (c) 2025 e-dynamics GmbH and affiliates
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.


// Priors shared by all models
k_std ~ normal(0,0.5 * t_scale);
m_n ~ normal(0.5,0.5);
delta_std ~ double_exponential(0, delta_scale * t_scale);
delta_std ~ normal(0, 7e-4 * t_scale);
beta_n ~ normal(0, beta_scale);