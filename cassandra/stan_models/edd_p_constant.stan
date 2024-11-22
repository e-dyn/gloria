data {
  int<lower=0> T;               // Number of time periods
  array[T] int<lower=0> y;      // Time series
  array[T] int N;               // Population size, vectorized form
}

parameters {
  real<lower=0, upper=1> p;     // Success probability
}

model {
  p ~ normal(0.5,1);
  
  // Likelihood
  y ~ binomial(N, p);
}