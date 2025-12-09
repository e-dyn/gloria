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
  real<lower=0> gamma;          // Scale on disperion proxy prior
  array[T] int capacity;        // Capacity
}

transformed data {
  // Data transformations shared by all models
  #include utilities/transformed_data.stan
  
  // Model specific data transformations  
  // Calculate dispersion prior scales
  // Note: Factor 1/6 is chosen such that the Prior is sensitive around 
  // kappa=0.5 for the default prior scale gamma=3.
  real<lower=0> gamma_scale = gamma / 6;
  
  vector[T] capacity_vec = to_vector(capacity);
  real eps = 1e-9;
}

parameters {
  // Model parameters shared by all models
  #include utilities/parameters.stan
  
  // Model specific parameters
  real<lower=0,upper=2> kappa;        // Dispersion proxy
}

transformed parameters {
  // Transformer parameters shared by all models
  #include utilities/transformed_parameters.stan
  
  vector[T] scale = 4*(capacity_vec-1)./(capacity_vec*kappa^2) - 1;         // Scale parameter for distribution

  vector[T] p = inv_logit(                      // Model success probability
      linked_offset 
      + linked_scale*(trend + X * beta)
  );
  p = fmin(fmax(p, eps), 1 - eps);              // ensure p is inside (0,1)
  // Relate p and scale to standard parameters for Beta-Binomial
  vector[T] a = p .* scale;
  vector[T] b = (1 - p) .* scale;
}

model {
  // Priors shared by all models
  #include utilities/priors.stan
  
  // Model specific priors
  kappa ~ exponential(gamma_scale);
  
  // Likelihood  
  for (n in 1:num_elements(y)) {
    y[n] ~ beta_binomial(capacity[n], a[n], b[n]);
  }
}

#include utilities/generated_quantities.stan