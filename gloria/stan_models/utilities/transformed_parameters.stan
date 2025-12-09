// Copyright (c) 2025 e-dynamics GmbH and affiliates
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// Transformer parameters shared by all models
vector[T] trend = linear_trend(
    k, m, delta,
    t, A, t_change
);