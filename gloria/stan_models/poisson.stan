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
  // Transformer parameters shared by all models
  #include utilities/transformed_parameters.stan
}

model {
  // Priors shared by all models
  #include utilities/priors.stan
  
  // Likelihood
  y ~ poisson_log_glm(
    X_n,
    linked_offset + linked_scale * trend,      // Denormalized trend
    linked_scale * beta_n                      // Denormalized regression coefficients
  );
}

#include utilities/generated_quantities.stan