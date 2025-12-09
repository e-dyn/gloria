// Copyright (c) 2025 e-dynamics GmbH and affiliates
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include utilities/functions.stan

data {
  int<lower=0> T;               // Number of time periods
  int<lower=0> S;               // Number of changepoints
  int<lower=0> K;               // Number of regressors
  real<lower=0> tau;            // Scale on changepoints prior
  array[T] int<lower=0> y;      // Time series
  vector[T] t;                  // Time as integer vector
  vector[S] t_change;           // Times of trend changepoints as integers
  matrix[T,K] X;                // Regressors
  vector[K] sigmas;             // Scale on seasonality prior
  array[T] int capacity;        // Capacity
  real linked_offset;           // Offset of linear model
  real linked_scale;            // Scale of linear model
}

transformed data {
  #include utilities/transformed_data.stan
}

parameters {
  real<lower=-0.5, upper=0.5> k;            // Base trend growth rate
  real<lower=0, upper=1> m;                 // Trend offset
  vector<lower=-1, upper=1>[S] delta;       // Trend rate adjustments
  vector<                                   // Regressor coefficients
    lower=-1/reg_scales,
    upper=1/reg_scales
  >[K] beta;  
  // Note: lower and upper bounds 1/reg_scales are chosen such that each 
  // regressor is able to bridge the entire range of the normalized linear 
  // model range [0,1]
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
  y ~ binomial_logit_glm(
    capacity,
    X,
    linked_offset + linked_scale * trend,    // Denormalized trend
    linked_scale * beta                      // Denormalized regression coefficients
  ); 
}