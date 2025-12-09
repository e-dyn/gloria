// Copyright (c) 2025 e-dynamics GmbH and affiliates
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

generated quantities {
  real k = k_std / t_scale;
  vector[S] delta = delta_std / t_scale;
}