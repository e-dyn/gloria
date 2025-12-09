// Copyright (c) 2025 e-dynamics GmbH and affiliates
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include utilities/functions.stan

data {
  // Input data shared by all models
  #include utilities/data.stan

  // Model specific input data
  array[T] int<lower=0> y;      // Time series
}

transformed data {
  // Data transformations shared by all models
  #include utilities/transformed_data.stan
}

parameters {
  // Model parameters shared by all models
  #include utilities/parameters.stan
}

transformed parameters {
  vector[T] trend = linear_trend(
      k, m, delta,
      t, A, t_change
  );
}

model {
  // Priors
  k ~ normal(0,0.5);
  m ~ normal(0.5,0.5);
  delta ~ double_exponential(0, delta_scale);
  // Note: Factor 0.072 is chosen such that with tau=3 the double_exponential
  // drops to 1% of its maximum value for delta_max = 1
  beta ~ normal(0, beta_scale);
  
  // Likelihood
  y ~ poisson_log_glm(
    X,
    linked_offset + linked_scale * trend,    // Denormalized trend
    linked_scale * beta                      // Denormalized regression coefficients
  );
}