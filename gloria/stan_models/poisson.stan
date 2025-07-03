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
}

transformed data {
  matrix[T, S] A = get_changepoint_matrix(t, t_change, T, S);
  
  // Get normalization parameters for linear model
  vector[T] y_real = to_vector(to_array_1d(y));       // Convert y to vector of real values
  vector[T] y_linked = log(y_real);                   // Apply link function
  real linked_offset = min(y_linked);                 // Offset of linear model
  real linked_scale = max(y_linked) - linked_offset;  // Scale of linear model
}

parameters {
  real<lower=-0.5, upper=0.5> k;                       // Base trend growth rate
  real<lower=0, upper=1> m;                       // Trend offset
  vector<lower=-0.5, upper=0.5>[S] delta;              // Trend rate adjustments
  vector<lower=-1, upper=1>[K] beta;               // Slope for y
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
  delta ~ double_exponential(0, 0.036*tau); // Chosen such that with tau=3 probability to get a delta on its bounds is ~1%
  beta ~ normal(0, 0.155*sigmas); // Chosen such that with sigma=3 probability to get a beta on its bounds is ~1%
  
  // Likelihood
  y ~ poisson_log_glm(
    X,
    linked_offset + linked_scale * trend,    // Denormalized trend
    linked_scale * beta                      // Denormalized regression coefficients
  );
}

generated quantities {
  real print_offset = linked_offset;
  real print_scale = linked_scale;
}