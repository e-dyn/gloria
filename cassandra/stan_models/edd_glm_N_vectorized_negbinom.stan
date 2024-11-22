data {
  int<lower=0> T;               // Number of time periods
  int<lower=1> K;               // Number of regressors
  array[T] int<lower=0> y;      // Time series
  matrix[T,K] X;                // Regressors
  array[T] int N;                   // Population size, vectorized form
}

parameters {
  vector[K] beta;               // Slope for y
  real alpha;                   // Intercept for y
  real<lower=0> phi;
}

model {
  // Priors
  beta ~ std_normal();
  alpha ~ std_normal();
  phi ~ std_normal();
  
  // Likelihood
  y ~ neg_binomial_2_log_glm(X, alpha, beta, phi);
}