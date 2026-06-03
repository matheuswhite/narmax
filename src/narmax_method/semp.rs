use crate::{
    narmax_method::{NarmaxMethod, common},
    narmax_model::{NarmaxModel, mse},
    regressor::Regressor,
};
use nalgebra::{DMatrix, DVector};

/// **Simulation Error Minimization with Pruning** (Piroddi & Spinelli, 2003).
///
/// Forward selection where the accept/reject decision is driven by the **free-run
/// simulation error** (MSSE) rather than the one-step ERR. ERR is used only to rank
/// which candidates to try first. After every accepted term a **pruning** pass removes
/// terms whose deletion does not worsen the MSSE, countering the greediness of forward
/// selection. NARX-only: the residual channel is not modeled.
#[derive(Clone, Copy)]
pub struct Semp {
    /// Hard cap on the number of selected terms.
    max_terms: usize,
    /// Minimum relative MSSE improvement to keep iterating.
    tol: f32,
    /// Number of top-ERR candidates trialled per forward step (`usize::MAX` = exact).
    top_k: usize,
}

impl Semp {
    pub fn new(max_terms: usize, tol: f32, top_k: usize) -> Self {
        Self {
            max_terms,
            tol,
            top_k,
        }
    }

    fn core_identify(
        &self,
        regressors: Vec<Regressor>,
        y: &[f32],
        u: &[f32],
        phi: DMatrix<f32>,
    ) -> NarmaxModel {
        let m_cols = phi.ncols();
        let n = phi.nrows();
        let k_min_phi = y.len() - n;
        let yv = DVector::from_fn(n, |i, _| y[k_min_phi + i]);

        let mut selected: Vec<usize> = Vec::new();
        let mut best_msse = f32::INFINITY;

        loop {
            /* ---- FORWARD STEP ---- */
            let remaining: Vec<usize> = (0..m_cols).filter(|j| !selected.contains(j)).collect();
            if remaining.is_empty() || selected.len() >= self.max_terms {
                break;
            }

            let ranked = rank_by_err(&phi, &selected, &yv, &remaining);

            let mut accepted: Option<(usize, f32)> = None;
            for &c in ranked.iter().take(self.top_k) {
                let mut trial = selected.clone();
                trial.push(c);
                let Some(theta) = ls_estimate(&phi, &trial, &yv) else {
                    continue;
                };
                let msse = msse_of(&regressors, &theta, &trial, y, u);
                if msse < best_msse {
                    accepted = Some((c, msse));
                    break;
                }
            }

            let Some((c, new_msse)) = accepted else {
                break; // no candidate reduces MSSE -> stop
            };

            let prev_best = best_msse;
            let rel_impr = (best_msse - new_msse) / best_msse.max(f32::EPSILON);
            selected.push(c);
            best_msse = new_msse;

            /* ---- PRUNING STEP (until stable) ---- */
            loop {
                let mut removed_any = false;
                let mut i = 0;
                while i < selected.len() {
                    if selected.len() <= 1 {
                        break;
                    }
                    let mut trial = selected.clone();
                    trial.remove(i);
                    if let Some(theta) = ls_estimate(&phi, &trial, &yv) {
                        let msse = msse_of(&regressors, &theta, &trial, y, u);
                        if msse <= best_msse {
                            selected = trial;
                            best_msse = msse;
                            removed_any = true;
                            continue; // re-examine same index (now the next term)
                        }
                    }
                    i += 1;
                }
                if !removed_any {
                    break;
                }
            }

            /* ---- STOP on small relative improvement ---- */
            if prev_best.is_finite() && rel_impr < self.tol {
                break;
            }
        }

        /* ---- FINAL ESTIMATE ---- */
        let theta = ls_estimate(&phi, &selected, &yv)
            .unwrap_or_else(|| DVector::zeros(selected.len()));
        let selected_regressors = selected
            .iter()
            .map(|&i| regressors[i].clone())
            .collect::<Vec<_>>();

        NarmaxModel {
            regressors: selected_regressors,
            theta: theta.iter().copied().collect(),
            phi: Some(phi),
            err: None,
            selected_indices: Some(selected),
        }
    }
}

/// Rank remaining candidates by ERR relative to the currently selected columns.
/// Builds an orthogonal basis of the selected columns (Gram-Schmidt), residualizes
/// each candidate against it, and orders by `g^2 * ||w||^2` (descending). The constant
/// `sigma2` scaling is dropped since it does not affect the ordering.
fn rank_by_err(
    phi: &DMatrix<f32>,
    selected: &[usize],
    yv: &DVector<f32>,
    remaining: &[usize],
) -> Vec<usize> {
    let mut basis: Vec<DVector<f32>> = Vec::new();
    let mut norms2: Vec<f32> = Vec::new();
    for &s in selected {
        let mut col = phi.column(s).clone_owned();
        for (wk, nk) in basis.iter().zip(&norms2) {
            let alpha = wk.dot(&col) / nk;
            col -= alpha * wk;
        }
        let nrm = col.norm_squared();
        if nrm > 1e-8 {
            basis.push(col);
            norms2.push(nrm);
        }
    }

    let mut scored: Vec<(f32, usize)> = remaining
        .iter()
        .map(|&j| {
            let mut col = phi.column(j).clone_owned();
            for (wk, nk) in basis.iter().zip(&norms2) {
                let alpha = wk.dot(&col) / nk;
                col -= alpha * wk;
            }
            let nrm = col.norm_squared();
            let score = if nrm > 1e-8 {
                let gg = col.dot(yv) / nrm;
                gg * gg * nrm
            } else {
                0.0
            };
            (score, j)
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(_, j)| j).collect()
}

/// Ordinary least squares of `y` on the Phi columns indexed by `m`, via the normal
/// equations. Returns `None` on an empty set, a singular system, or non-finite theta.
fn ls_estimate(phi: &DMatrix<f32>, m: &[usize], y: &DVector<f32>) -> Option<DVector<f32>> {
    if m.is_empty() {
        return None;
    }
    let sub = phi.select_columns(m.iter());
    let xtx = sub.transpose() * &sub;
    let xty = sub.transpose() * y;
    let lu = xtx.lu();
    if !lu.is_invertible() {
        return None;
    }
    let theta = lu.solve(&xty)?;
    if theta.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(theta)
}

/// Mean square simulation (free-run) error on the training data for the model defined
/// by the index set `m` and its LS coefficients. Divergent models map to `+inf`.
fn msse_of(
    regressors: &[Regressor],
    theta: &DVector<f32>,
    m: &[usize],
    y: &[f32],
    u: &[f32],
) -> f32 {
    let regs = m.iter().map(|&i| regressors[i].clone()).collect::<Vec<_>>();
    let model = NarmaxModel {
        regressors: regs,
        theta: theta.iter().copied().collect(),
        phi: None,
        err: None,
        selected_indices: None,
    };
    let k_min = model.max_lag();
    let y_hat = model.simulate_free_run(y, u);
    let v = mse(&y[k_min..], &y_hat[k_min..]);
    if v.is_finite() { v } else { f32::INFINITY }
}

impl NarmaxMethod for Semp {
    fn identify(self, regressors: Vec<Regressor>, y: &[f32], u: &[f32]) -> NarmaxModel {
        let phi = common::build_initial_phi(&regressors, y, u);
        self.core_identify(regressors, y, u, phi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidates() -> Vec<Regressor> {
        let atoms = vec![
            Regressor::new("y", 1),
            Regressor::new("y", 2),
            Regressor::new("u", 1),
            Regressor::new("u", 2),
        ];
        let mut all = atoms.clone();
        for a in &atoms {
            for b in &atoms {
                let p = a * b;
                if !all.contains(&p) {
                    all.push(p);
                }
            }
        }
        all
    }

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

    #[test]
    fn semp_recovers_known_linear_system() {
        let (u, y) = synthetic_data(400);
        let regs = candidates();

        let model = Semp::new(8, 1e-4, 8).identify(regs, &y, &u);

        let y1 = Regressor::new("y", 1);
        let u1 = Regressor::new("u", 1);

        let pos_y1 = model.regressors.iter().position(|r| *r == y1);
        let pos_u1 = model.regressors.iter().position(|r| *r == u1);

        assert!(pos_y1.is_some(), "SEMP did not select y(k-1)");
        assert!(pos_u1.is_some(), "SEMP did not select u(k-1)");

        assert!(
            (model.theta[pos_y1.unwrap()] - 0.5).abs() < 1e-2,
            "theta(y(k-1)) = {} (expected ~0.5)",
            model.theta[pos_y1.unwrap()]
        );
        assert!(
            (model.theta[pos_u1.unwrap()] - 0.3).abs() < 1e-2,
            "theta(u(k-1)) = {} (expected ~0.3)",
            model.theta[pos_u1.unwrap()]
        );

        // All recovered coefficients must be finite (divergence-rejection path works).
        assert!(model.theta.iter().all(|v| v.is_finite()));
    }
}
