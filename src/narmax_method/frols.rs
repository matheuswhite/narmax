use crate::{narmax_method::NarmaxMethod, narmax_model::NarmaxModel, regressor::Regressor};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub struct Frols {
    rho: f32,
    l_max: usize,
}

impl Frols {
    pub fn new(rho: f32, l_max: usize) -> Self {
        Self { rho, l_max }
    }

    fn build_initial_phi(regressors: &[Regressor], y: &[f32], u: &[f32]) -> DMatrix<f32> {
        let mut k_min = 0;
        for r in regressors {
            for s in r.terms() {
                if s.index() > k_min {
                    k_min = s.index();
                }
            }
        }

        let samples = HashMap::from([("y", y), ("u", u)]);

        let n = y.len() - k_min;
        let m = regressors.len();

        DMatrix::from_fn(n, m, |i, j| {
            regressors[j].eval_at(k_min + i, &samples).unwrap()
        })
    }

    fn build_initial_phi_with_error(
        regressors: &[Regressor],
        y: &[f32],
        u: &[f32],
        e: &[f32],
    ) -> DMatrix<f32> {
        let mut k_min = 0;
        for r in regressors {
            for s in r.terms() {
                if s.index() > k_min {
                    k_min = s.index();
                }
            }
        }

        let samples = HashMap::from([("y", y), ("u", u), ("e", e)]);

        let n = y.len() - k_min;
        let m = regressors.len();

        DMatrix::from_fn(n, m, |i, j| {
            regressors[j].eval_at(k_min + i, &samples).unwrap()
        })
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

        /* Build Matrix and Vectors */
        let mut w = DMatrix::zeros(n, l_max);
        let mut norms2 = DVector::zeros(l_max);
        let mut g = DVector::zeros(l_max);
        let mut err = DVector::zeros(l_max);
        let mut a = DMatrix::zeros(l_max, l_max);
        let mut selected = Vec::with_capacity(l_max);
        let mut remaining = (0..m).collect::<Vec<_>>();
        let mut current_l = 0_usize;

        'main_loop: for _ in 0..l_max {
            let w_view = w.columns(0, current_l);
            let norms2_view = norms2.rows(0, current_l);

            let mut best_j = None;
            let mut best_err = f32::NEG_INFINITY;
            let mut best_w_cand = None;
            let mut best_g = 0.0;
            let mut best_w_norm2 = 0.0;
            let mut best_alpha = None;

            'inner_loop: for &j in &remaining {
                let phi_j = phi.column(j);

                let (w_cand, alpha) = if current_l == 0 {
                    let alpha = None;
                    let w_cand = phi_j.clone_owned();

                    (w_cand, alpha)
                } else {
                    let proj = w_view.tr_mul(&phi_j);
                    let alpha = proj.component_div(&norms2_view);
                    let w_cand = &phi_j - &(&w_view * &alpha);

                    (w_cand, Some(alpha))
                };

                let w_norm2 = w_cand.norm_squared();
                if w_norm2 < 1e-8 {
                    continue 'inner_loop;
                }

                let g_cand = w_cand.dot(&y) / w_norm2;
                let err_cand = g_cand * g_cand * w_norm2 / sigma2;

                if err_cand > best_err {
                    best_err = err_cand;
                    best_j = Some(j);
                    best_w_cand = Some(w_cand);
                    best_g = g_cand;
                    best_w_norm2 = w_norm2;
                    best_alpha = alpha;
                }
            }

            if best_j.is_none() {
                break 'main_loop;
            }

            /* COMMIT */
            w.set_column(current_l, &best_w_cand.unwrap());
            norms2[current_l] = best_w_norm2;
            g[current_l] = best_g;
            err[current_l] = best_err;

            if let Some(b_alpha) = best_alpha.as_ref() {
                for i in 0..current_l {
                    a[(i, current_l)] = b_alpha[i];
                }
            }
            a[(current_l, current_l)] = 1.0;

            selected.push(best_j.unwrap());
            remaining.retain(|&idx| idx != best_j.unwrap());
            current_l += 1;

            /* STOP */
            let esr = 1.0 - err.rows(0, current_l).sum();
            if esr < self.rho {
                break 'main_loop;
            }

            if remaining.is_empty() {
                break 'main_loop;
            }
        }

        /* Recover Theta from original space */
        let mut theta = DVector::zeros(current_l);
        for i in (0..current_l).rev() {
            let mut sum = 0.0;
            for k in (i + 1)..current_l {
                sum += a[(i, k)] * theta[k];
            }
            theta[i] = g[i] - sum;
        }

        /* Build and return model */
        let mut selected_regressors = vec![];
        for &idx in &selected {
            selected_regressors.push(regressors[idx].clone());
        }

        NarmaxModel {
            regressors: selected_regressors,
            theta: theta.iter().copied().collect(),
            phi: Some(phi),
            err: Some(err.rows(0, current_l).iter().cloned().collect()),
            selected_indices: Some(selected),
        }
    }
}

impl NarmaxMethod for Frols {
    fn identify(self, regressors: Vec<Regressor>, y: &[f32], u: &[f32]) -> NarmaxModel {
        let phi = Self::build_initial_phi(&regressors, y, u);
        self.core_identify(regressors, y, phi)
    }

    fn identify_with_error(
        self,
        regressors: Vec<Regressor>,
        y: &[f32],
        u: &[f32],
        e: &[f32],
    ) -> NarmaxModel {
        let phi = Self::build_initial_phi_with_error(&regressors, y, u, e);
        self.core_identify(regressors, y, phi)
    }
}
