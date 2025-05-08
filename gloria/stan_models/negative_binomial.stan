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
  array[T] int<lower=0> y;      // Time series
  vector[T] t;                  // Time as integer vector
  vector[S] t_change;           // Times of trend changepoints as integers
  matrix[T,K] X;                // Regressors
  vector[K] sigmas;             // Scale on seasonality prior
  vector[K] s_a;                // Indicator of additive features
  vector[K] s_m;                // Indicator of multiplicative features
  real scale_est;
}

transformed data {
  matrix[T, S] A = get_changepoint_matrix(t, t_change, T, S);
  matrix[T, K] X_sa = X .* rep_matrix(s_a', T);
  matrix[T, K] X_sm = X .* rep_matrix(s_m', T);
}

parameters {
  real k;                       // Base trend growth rate
  real m;                       // Trend offset
  vector[S] delta;              // Trend rate adjustments
  vector[K] beta;               // Slope for y
  real<lower = -1> scale_raw;
}

transformed parameters {
  vector[T] trend;
  trend = linear_trend(k, m, delta, t, A, t_change);
  real<lower = 0> scale = (scale_raw+1)*scale_est;
}

model {
  // Priors
  k ~ normal(0, 5);
  m ~ normal(0, 5);
  delta ~ double_exponential(0, tau);
  beta ~ normal(0, sigmas);
  scale_raw ~ std_normal();
  
  // Likelihood
  for (n in 1:num_elements(y)) {
    real eta_n;
    eta_n = trend[n] * (1 + X_sm[n] * beta) + X_sa[n] * beta;
    y[n] ~ neg_binomial_2_log(eta_n, scale);
  }
}