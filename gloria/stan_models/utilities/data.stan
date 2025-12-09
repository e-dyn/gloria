// Copyright (c) 2025 e-dynamics GmbH and affiliates
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// Input data shared by all models
int<lower=0> T;               // Number of time periods
int<lower=0> S;               // Number of changepoints
int<lower=0> K;               // Number of regressors
real<lower=0> tau;            // Scale on changepoints prior
vector[T] t;                  // Time as integer vector
vector[S] t_change;           // Times of trend changepoints as integers
matrix[T,K] X;                // Regressors
vector[K] sigmas;             // Scale on seasonality prior
real linked_offset;           // Offset of linear model
real linked_scale;            // Scale of linear model