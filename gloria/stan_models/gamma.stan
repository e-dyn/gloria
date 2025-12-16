// Copyright (c) 2025 e-dynamics GmbH and affiliates
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include utilities/functions.stan

data {
  // Input data shared by all models
  #include utilities/data.stan

  // Model specific input data
  array[T] real<lower=0> y;     // Time series
  real variance_max;            // Upper bound on the variance
  real<lower=0> gamma;          // Scale on disperion proxy prior  
}

transformed data {
  // Data transformations shared by all models
  #include utilities/transformed_data.stan
  
  // Model specific data transformations  
  // Calculate dispersion prior scales
  // Note: Factor 1/6 is chosen such that the Prior is sensitive around 
  // kappa=0.5 for the default prior scale gamma=3.
  real<lower=0> gamma_scale = gamma / 6;
  
  // Parameters for dispersion scale
  vector[T] y_real = to_vector(y);                    // Convert y to vector of real values
  real mu_mean = mean(y_real);                        // An estimate for the mean expectation value
}

parameters {
  // Model parameters shared by all models
  #include utilities/parameters.stan
  
  // Model specific parameters
  real<lower=0, upper=2> kappa;               // Dispersion proxy
}

transformed parameters {
  // Transformer parameters shared by all models
  #include utilities/transformed_parameters.stan
  
  real scale = mu_mean / (variance_max*kappa^2);  // Scale parameter for distribution
  
  vector[T] eta = scale * exp(                // Denormalization if linear model
      linked_offset 
      + linked_scale*(trend + X_n * beta_n)
  );
}

model {
  // Priors shared by all models
  #include utilities/priors.stan
  
  // Model specific priors
  kappa ~ exponential(gamma_scale);
  
  // Likelihood
  for (n in 1:num_elements(y)) {
    y[n] ~ gamma(eta[n], scale);
  }
}

#include utilities/generated_quantities.stan