// Copyright (c) 2025 e-dynamics GmbH and affiliates
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include utilities/functions.stan

data {
  #include utilities/data.stan

  // Model specific input data
  array[T] real<lower=0> y;     // Time series
  real variance_max;            // Upper bound on the variance
  real<lower=0> gamma;          // Scale on disperion proxy prior
}

transformed data {
  #include utilities/transformed_data.stan
  
  // Calculate dispersion prior scales
  // Note: Factor 1/6 is chosen such that the Prior is sensitive around 
  // kappa=0.5 for the default prior scale gamma=3.
  real<lower=0> gamma_scale = gamma / 6;
  
  // Parameters for dispersion scale
  real eps = 1e-9;
  vector[T] y_real = to_vector(y);                    // Convert y to vector of real values
  real mu_mean = mean(y_real);                        // An estimate for the mean expectation value
  real kappa_max = fmin(mu_mean * (1-mu_mean) / variance_max -eps, 2.);
}

parameters {
  real<lower=-0.5, upper=0.5> k;              // Base trend growth rate
  real<lower=0, upper=1> m;                   // Trend offset
  vector<lower=-1, upper=1>[S] delta;         // Trend rate adjustments
  vector<                                     // Regressor coefficients
    lower=-1/reg_scales,
    upper=1/reg_scales
  >[K] beta;  
  // Note: lower and upper bounds 1/reg_scales are chosen such that each 
  // regressor is able to bridge the entire range of the normalized linear 
  // model range [0,1]
  real<lower=0, upper=kappa_max> kappa;          // Dispersion proxy
}

transformed parameters {
  vector[T] trend = linear_trend(k, m, delta, t, A, t_change);
  // Scale parameter for distribution
  real scale = mu_mean * (1-mu_mean) / (variance_max * kappa^2)-1;
  // Expectation value of Beta-distribution
  vector[T] mu = inv_logit(                // Denormalization if linear model
      linked_offset 
      + linked_scale*(trend + X * beta)
  );
  mu = fmin(fmax(mu, eps), 1 - eps); // ensure mu is inside (0,1)
}

model {
  // Priors
  k ~ normal(0,0.5);
  m ~ normal(0.5,0.5);
  delta ~ double_exponential(0, delta_scale);
  // Note: Factor 0.072 is chosen such that with tau=3 the double_exponential
  // drops to 1% of its maximum value for delta_max = 1
  beta ~ normal(0, beta_scale);
  kappa ~ exponential(gamma_scale);
  
  // Likelihood
  for (n in 1:num_elements(y)) {
    y[n] ~ beta_proportion(mu[n], scale);
  }
}