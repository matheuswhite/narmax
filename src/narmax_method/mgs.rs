use crate::{
    narmax_method::{NarmaxMethod, common},
    narmax_model::NarmaxModel,
    regressor::Regressor,
};
use nalgebra::{DMatrix, DVector};

/// FROLS variant using **Modified Gram-Schmidt** orthogonalization.
///
/// Same ERR selection criterion and stop criteria as [`super::frols::Frols`], but
/// instead of re-projecting the original column against the whole selected block at
/// every step (classical GS), it **deflates the candidate matrix in place**: after a
/// column is committed, its projection is subtracted from every remaining candidate.
/// This is numerically more stable (relevant under `f32` + near-colinearity) and in
/// exact arithmetic produces the same model as FROLS.
#[derive(Clone, Copy)]
pub struct Mgs {
    rho: f32,
    l_max: usize,
}

impl Mgs {
    pub fn new(rho: f32, l_max: usize) -> Self {
        Self { rho, l_max }
    }

    fn core_identify(
        &self,
        regressors: Vec<Regressor>,
        y: &[f32],
        phi: DMatrix<f32>,
    ) -> NarmaxModel {
        /* Setup */
        let m = phi.ncols();
        let n = phi.nrows();
        let k_min = y.len() - n;
        let y = DVector::from_fn(n, |i, _| y[k_min + i]);
        let sigma2 = (y.transpose() * &y)[(0, 0)];
        let l_max = self.l_max.min(m);

        /* Working state */
        let mut p = phi.clone(); // deflated in place
        let mut norms2 = DVector::zeros(l_max);
        let mut g = DVector::zeros(l_max);
        let mut err = DVector::zeros(l_max);
        let mut a = DMatrix::zeros(l_max, l_max);
        // Per ORIGINAL candidate index: deflation coefficients in step order.
        let mut alphas: Vec<Vec<f32>> = vec![Vec::new(); m];
        let mut selected = Vec::with_capacity(l_max);
        let mut remaining = (0..m).collect::<Vec<_>>();
        let mut current_l = 0_usize;

        'main_loop: for _ in 0..l_max {
            /* (1) Score remaining candidates using the CURRENT deflated columns,
                   which are already orthogonal to all previously selected ones. */
            let mut best_j = None;
            let mut best_err = f32::NEG_INFINITY;
            let mut best_g = 0.0;
            let mut best_norm2 = 0.0;

            for &j in &remaining {
                let pj = p.column(j);
                let norm2 = pj.norm_squared();
                if norm2 < 1e-8 {
                    continue;
                }

                let g_cand = pj.dot(&y) / norm2;
                let err_cand = g_cand * g_cand * norm2 / sigma2;

                if err_cand > best_err {
                    best_err = err_cand;
                    best_j = Some(j);
                    best_g = g_cand;
                    best_norm2 = norm2;
                }
            }

            let Some(s) = best_j else {
                break 'main_loop;
            };

            /* (2) Commit column s as the current_l-th selected term. */
            let w_s = p.column(s).clone_owned();
            norms2[current_l] = best_norm2;
            g[current_l] = best_g;
            err[current_l] = best_err;

            for i in 0..current_l {
                a[(i, current_l)] = alphas[s][i];
            }
            a[(current_l, current_l)] = 1.0;

            selected.push(s);
            remaining.retain(|&idx| idx != s);
            current_l += 1;

            /* STOP */
            let esr = 1.0 - err.rows(0, current_l).sum();
            if esr < self.rho {
                break 'main_loop;
            }
            if remaining.is_empty() {
                break 'main_loop;
            }

            /* (3) Deflate every remaining candidate against w_s, in place, and
                   record the coefficient for the eventual A-matrix column. */
            for &j in &remaining {
                let beta = w_s.dot(&p.column(j)) / best_norm2;
                let new_col = &p.column(j) - beta * &w_s;
                p.set_column(j, &new_col);
                alphas[j].push(beta);
            }
        }

        /* Recover Theta from original space (same back-substitution as FROLS). */
        let theta = common::recover_theta(&a, &g, current_l);

        /* Build and return model */
        let selected_regressors = selected
            .iter()
            .map(|&idx| regressors[idx].clone())
            .collect::<Vec<_>>();

        NarmaxModel {
            regressors: selected_regressors,
            theta: theta.iter().copied().collect(),
            phi: Some(phi),
            err: Some(err.rows(0, current_l).iter().cloned().collect()),
            selected_indices: Some(selected),
        }
    }
}

impl NarmaxMethod for Mgs {
    fn identify(self, regressors: Vec<Regressor>, y: &[f32], u: &[f32]) -> NarmaxModel {
        let phi = common::build_initial_phi(&regressors, y, u);
        self.core_identify(regressors, y, phi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::narmax_method::frols::Frols;

    /// Degree-2 candidate set over [y(k-1), y(k-2), u(k-1), u(k-2)] (14 terms),
    /// matching `build_regressors([("y",2),("u",2)], 2)`.
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

    /// Deterministic well-conditioned linear system: y[k] = 0.5 y[k-1] + 0.3 u[k-1] + tiny noise.
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
    fn mgs_equivalent_to_frols() {
        let (u, y) = synthetic_data(400);
        let regs = candidates();

        let mf = Frols::new(1e-3, 8).identify(regs.clone(), &y, &u);
        let mm = Mgs::new(1e-3, 8).identify(regs.clone(), &y, &u);

        // Same selected regressors, in the same selection order.
        assert_eq!(
            mf.regressors.len(),
            mm.regressors.len(),
            "FROLS and MGS selected a different number of terms"
        );
        for (a, b) in mf.regressors.iter().zip(&mm.regressors) {
            assert_eq!(a, b, "selection order differs: {a} vs {b}");
        }

        // Thetas match within a loose f32 tolerance.
        for (tf, tm) in mf.theta.iter().zip(&mm.theta) {
            assert!(
                (tf - tm).abs() < 1e-3,
                "theta mismatch: frols={tf} mgs={tm}"
            );
        }
    }
}
