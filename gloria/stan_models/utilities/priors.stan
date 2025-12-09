// Copyright (c) 2025 e-dynamics GmbH and affiliates
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.


// Priors shared by all models
k ~ normal(0,0.5);
m ~ normal(0.5,0.5);
delta ~ double_exponential(0, delta_scale);
beta ~ normal(0, beta_scale);