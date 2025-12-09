// Copyright (c) 2025 e-dynamics GmbH and affiliates
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include utilities/functions.stan

data {
  #include utilities/data.stan

  // Model specific input data
  vector[T] y;                  // Time series 
  real variance_max;            // Upper bound on the variance
  real<lower=0> gamma;          // Scale on disperion proxy prior
}

transformed data {
  #include utilities/transformed_data.stan
  
  // Calculate dispersion prior scales
  // Note: Factor 1/6 is chosen such that the Prior is sensitive around 
  // kappa=0.5 for the default prior scale gamma=3.
  real<lower=0> gamma_scale = gamma / 6;
}

parameters {
  real<lower=-0.5, upper=0.5> k;             // Base trend growth rate
  real<lower=0, upper=1> m;                  // Trend offset
  vector<lower=-1, upper=1>[S] delta;        // Trend rate adjustments
  vector<                                    // Regressor coefficients
    lower=-1/reg_scales,
    upper=1/reg_scales
  >[K] beta;  
  // Note: lower and upper bounds 1/reg_scales are chosen such that each 
  // regressor is able to bridge the entire range of the normalized linear 
  // model range [0,1]
  real<lower=0, upper=2> kappa;               // Dispersion proxy
}

transformed parameters {
  vector[T] trend = linear_trend(
      k, m, delta,
      t, A, t_change
  );
  real scale = sqrt(variance_max) * kappa; // Scale parameter for distribution
}

model {
  // Priors
  k ~ normal(0,0.5);
  m ~ normal(0.5,0.5);
  delta ~ double_exponential(0, delta_scale);
  beta ~ normal(0, beta_scale);
  kappa ~ exponential(gamma_scale);
  
  // Likelihood
  y ~ normal_id_glm(
    X,
    linked_offset + linked_scale * trend,    // Denormalized trend
    linked_scale * beta,                     // Denormalized regression coefficients
    scale
  );
}
