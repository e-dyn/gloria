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
  real linked_offset;           // Offset of linear model
  real linked_scale;            // Scale of linear model
}

transformed data {
  matrix[T, S] A = get_changepoint_matrix(t, t_change, T, S);
  
  // Find regressor-wise scales
  vector[K] reg_scales;
  for (j in 1:K) {
    reg_scales[j] = max(X[, j]) - min(X[, j]);
  }
  
  // Scaling factor for beta-prior to guarantee that it drops to 1% of its
  // maximum value at beta_max = 1/reg_scales for sigma = 3
  vector[K] f_beta = inv_sqrt(-2*log(0.01)*reg_scales^2) / 3;
  
  // Parameters for dispersion scale
  vector[T] y_real = to_vector(y);                    // Convert y to vector of real values
  real mu_mean = mean(y_real);                        // An estimate for the mean expectation value
  real data_range = max(y_real) - min(y_real);        // Full data range
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
  real<lower=0, upper=2> kappa;               // Dispersion proxy
}

transformed parameters {
  vector[T] trend = linear_trend(
      k, m, delta,
      t, A, t_change
  );
  real scale = mu_mean * inv_square(data_range*kappa);  // Scale parameter for distribution
  
  vector[T] eta = scale * exp(                // Denormalization if linear model
      linked_offset 
      + linked_scale*(trend + X * beta)
  );
}

model {
  // Priors
  k ~ normal(0,0.5);
  m ~ normal(0.5,0.5);
  delta ~ double_exponential(0, 0.072*tau);
  // Note: Factor 0.072 is chosen such that with tau=3 the double_exponential
  // drops to 1% of its maximum value for delta_max = 1
  beta ~ normal(0, f_beta.*sigmas);
  kappa ~ std_normal();
  
  // Likelihood
  for (n in 1:num_elements(y)) {
    y[n] ~ gamma(eta[n], scale);
  }
}


generated quantities {
  real print_offset = linked_offset;
  real print_scale = linked_scale;
}