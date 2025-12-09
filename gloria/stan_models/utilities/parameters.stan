// Copyright (c) 2025 e-dynamics GmbH and affiliates
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// Model parameters shared by all models
real<lower=-0.5, upper=0.5> k;        // Base trend growth rate
real<lower=0, upper=1> m;             // Trend offset
vector<lower=-1, upper=1>[S] delta;   // Trend rate adjustments
vector[K] beta;                       // Regressor coefficients
