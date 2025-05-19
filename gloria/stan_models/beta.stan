functions {
  matrix get_changepoint_matrix(vector t, vector t_change, int T, int S) {
    // Assumes t and t_change are sorted.
    matrix[T, S] A;
    row_vector[S] a_row;
    int cp_idx;

    // Start with an empty matrix.
    A = rep_matrix(0, T, S);
    a_row = rep_row_vector(0, S);
    cp_idx = 1;

    // Fill in each row of A.
    for (i in 1:T) {
      while ((cp_idx <= S) && (t[i] >= t_change[cp_idx])) {
        a_row[cp_idx] = 1;
        cp_idx = cp_idx + 1;
      }
      A[i] = a_row;
    }
    return A;
  }
  
  // Linear trend function
  vector linear_trend(
    real k,
    real m,
    vector delta,
    vector t,
    matrix A,
    vector t_change
  ) {
    return (k + A * delta) .* t + (m + A * (-t_change .* delta));
  }
}

data {
  int<lower=0> T;               // Number of time periods
  int<lower=0> S;               // Number of changepoints
  int<lower=0> K;               // Number of regressors
  real<lower=0> tau;            // Scale on changepoints prior
  array[T] real<lower=0> y;     // Time series
  vector[T] t;                  // Time as integer vector
  vector[S] t_change;           // Times of trend changepoints as integers
  matrix[T,K] X;                // Regressors
  vector[K] sigmas;             // Scale on seasonality prior
}

transformed data {
  matrix[T, S] A = get_changepoint_matrix(t, t_change, T, S);
  real eps = 1e-9;
}

parameters {
  real k;                       // Base trend growth rate
  real m;                       // Trend offset
  vector[S] delta;              // Trend rate adjustments
  vector[K] beta;               // Slope for y
  real<lower=0> kappa;          // Dispersion proxy
}

transformed parameters {
  vector[T] trend = linear_trend(k, m, delta, t, A, t_change);
  // Scale parameter for distribution
  real<lower=0> scale = inv_square(kappa)-1;
  // Expectation value of Beta-distribution
  vector[T] mu = inv_logit(trend + X * beta);
  mu = fmin(fmax(mu, eps), 1 - eps); // ensure mu is inside (0,1)
}

model {
  // Priors
  k ~ normal(0, 5);
  m ~ normal(0, 5);
  delta ~ double_exponential(0, tau);
  beta ~ normal(0, sigmas);
  kappa ~ exponential(1.0);
  
  // Likelihood
  for (n in 1:num_elements(y)) {
    y[n] ~ beta_proportion(mu[n], scale);
  }
}