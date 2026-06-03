use crate::{
    narmax_model::{NarmaxModel, max_lag, mse},
    regressor::Regressor,
};

/// Refine the coefficient vector `theta0` by **minimizing the mean square free-run
/// (simulation) error** on the training data, using a dependency-free Nelder-Mead
/// simplex search.
///
/// This mirrors the reference Python notebook's SEMP refinement (`scipy.optimize.minimize`
/// over a simulation-error objective). Unlike the closed-form least-squares estimate, the
/// objective here is the full multi-step free-run error as a function of `theta`, so every
/// evaluation runs a complete free-run simulation — which is exactly why this step is
/// orders of magnitude more expensive than LS.
///
/// The model structure (`regressors`) is fixed; only the coefficients are optimized.
pub fn refine_simulation_error(
    regressors: &[Regressor],
    theta0: &[f32],
    y: &[f32],
    u: &[f32],
    max_iter: usize,
) -> Vec<f32> {
    let p = theta0.len();
    if p == 0 {
        return vec![];
    }

    let k_min = max_lag(regressors);

    // Objective: MSSE(theta) on the training set. Divergent models map to a large value.
    let cost = |theta: &[f32]| -> f32 {
        let model = NarmaxModel {
            regressors: regressors.to_vec(),
            theta: theta.to_vec(),
            phi: None,
            err: None,
            selected_indices: None,
        };
        let y_hat = model.simulate_free_run(y, u);
        let v = mse(&y[k_min..], &y_hat[k_min..]);
        if v.is_finite() { v } else { f32::MAX }
    };

    /* Initial simplex: theta0 plus one perturbed vertex per dimension. */
    let mut simplex: Vec<Vec<f32>> = Vec::with_capacity(p + 1);
    simplex.push(theta0.to_vec());
    for i in 0..p {
        let mut v = theta0.to_vec();
        let step = if theta0[i].abs() > 1e-6 {
            0.05 * theta0[i]
        } else {
            0.05
        };
        v[i] += step;
        simplex.push(v);
    }
    let mut fvals: Vec<f32> = simplex.iter().map(|v| cost(v)).collect();

    /* Nelder-Mead coefficients (reflection, expansion, contraction, shrink). */
    let alpha = 1.0;
    let gamma = 2.0;
    let rho = 0.5;
    let sigma = 0.5;

    for _ in 0..max_iter {
        // Order vertices by cost (ascending).
        let mut order: Vec<usize> = (0..=p).collect();
        order.sort_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let best = order[0];
        let second_worst = order[p - 1];
        let worst = order[p];

        // Convergence: simplex costs collapsed.
        if (fvals[worst] - fvals[best]).abs() <= 1e-12 {
            break;
        }

        // Centroid of all vertices except the worst.
        let mut centroid = vec![0.0_f32; p];
        for &o in order.iter().take(p) {
            for j in 0..p {
                centroid[j] += simplex[o][j];
            }
        }
        for c in centroid.iter_mut() {
            *c /= p as f32;
        }

        // Reflection.
        let reflected: Vec<f32> = (0..p)
            .map(|j| centroid[j] + alpha * (centroid[j] - simplex[worst][j]))
            .collect();
        let f_ref = cost(&reflected);

        if f_ref < fvals[best] {
            // Expansion.
            let expanded: Vec<f32> = (0..p)
                .map(|j| centroid[j] + gamma * (reflected[j] - centroid[j]))
                .collect();
            let f_exp = cost(&expanded);
            if f_exp < f_ref {
                simplex[worst] = expanded;
                fvals[worst] = f_exp;
            } else {
                simplex[worst] = reflected;
                fvals[worst] = f_ref;
            }
        } else if f_ref < fvals[second_worst] {
            simplex[worst] = reflected;
            fvals[worst] = f_ref;
        } else {
            // Contraction (toward the worst vertex).
            let contracted: Vec<f32> = (0..p)
                .map(|j| centroid[j] + rho * (simplex[worst][j] - centroid[j]))
                .collect();
            let f_con = cost(&contracted);
            if f_con < fvals[worst] {
                simplex[worst] = contracted;
                fvals[worst] = f_con;
            } else {
                // Shrink all vertices toward the best.
                let b = simplex[best].clone();
                for &o in order.iter().skip(1) {
                    for j in 0..p {
                        simplex[o][j] = b[j] + sigma * (simplex[o][j] - b[j]);
                    }
                    fvals[o] = cost(&simplex[o]);
                }
            }
        }
    }

    // Return the best vertex found.
    let mut best = 0;
    for i in 1..=p {
        if fvals[i] < fvals[best] {
            best = i;
        }
    }
    simplex[best].clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_data(n: usize) -> (Vec<f32>, Vec<f32>) {
        let mut state: u32 = 12345;
        let mut rng = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        };
        let mut u = vec![0.0f32; n];
        let mut y = vec![0.0f32; n];
        for k in 0..n {
            u[k] = rng();
        }
        for k in 2..n {
            y[k] = 0.5 * y[k - 1] + 0.3 * u[k - 1] + 0.001 * rng();
        }
        (u, y)
    }

    /// Refinement must not increase the training simulation error: the returned theta's
    /// MSSE is at most that of the initial (deliberately perturbed) theta.
    fn msse(regs: &[Regressor], theta: &[f32], y: &[f32], u: &[f32]) -> f32 {
        let model = NarmaxModel {
            regressors: regs.to_vec(),
            theta: theta.to_vec(),
            phi: None,
            err: None,
            selected_indices: None,
        };
        let k_min = max_lag(regs);
        mse(&y[k_min..], &model.simulate_free_run(y, u)[k_min..])
    }

    #[test]
    fn refine_does_not_worsen_simulation_error() {
        let (u, y) = synthetic_data(300);
        let regs = vec![Regressor::new("y", 1), Regressor::new("u", 1)];

        // Start from a deliberately off initial guess.
        let theta0 = vec![0.2_f32, 0.1_f32];
        let initial = msse(&regs, &theta0, &y, &u);

        let theta_ref = refine_simulation_error(&regs, &theta0, &y, &u, 500);
        let refined = msse(&regs, &theta_ref, &y, &u);

        assert!(
            refined <= initial + 1e-9,
            "refinement worsened MSSE: {initial} -> {refined}"
        );
        // It should also move the coefficients toward the true (0.5, 0.3) system.
        assert!((theta_ref[0] - 0.5).abs() < 0.1, "theta_y1 = {}", theta_ref[0]);
        assert!((theta_ref[1] - 0.3).abs() < 0.1, "theta_u1 = {}", theta_ref[1]);
    }
}
