// Copyright (c) 2025 e-dynamics GmbH and affiliates
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// Transformer parameters shared by all models

// Rescaled intercept using standardized time
real m_std = m_n + (k_std * t_center) / t_scale;

vector[T] trend = linear_trend(
    k_std, m_std, delta_std,
    t_std, A, t_change_std
);