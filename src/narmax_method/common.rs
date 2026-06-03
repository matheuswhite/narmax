use crate::regressor::Regressor;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// Max lag over ALL candidate regressors' symbols (the Phi row offset `k_min`).
///
/// Distinct from [`crate::narmax_model::max_lag`] applied to a *selected* model:
/// here it is computed over the full candidate set to size the regression matrix.
pub fn k_min_of(regressors: &[Regressor]) -> usize {
    let mut k_min = 0;
    for r in regressors {
        for s in r.terms() {
            if s.index() > k_min {
                k_min = s.index();
            }
        }
    }
    k_min
}

/// Build the initial regression matrix Phi (n x m) from the candidate set, using
/// the `y` and `u` channels.
pub fn build_initial_phi(regressors: &[Regressor], y: &[f32], u: &[f32]) -> DMatrix<f32> {
    let k_min = k_min_of(regressors);
    let samples = HashMap::from([("y", y), ("u", u)]);

    let n = y.len() - k_min;
    let m = regressors.len();

    DMatrix::from_fn(n, m, |i, j| {
        regressors[j].eval_at(k_min + i, &samples).unwrap()
    })
}

/// Recover `theta` in the ORIGINAL basis by back-substitution on the upper-triangular
/// matrix `a` (in selection order, unit diagonal) and the orthogonal gains `g`.
///
/// Solves `A theta = g` where `A` is `l x l` upper triangular: shared by the
/// orthogonal-least-squares methods (FROLS, MGS).
pub fn recover_theta(a: &DMatrix<f32>, g: &DVector<f32>, l: usize) -> DVector<f32> {
    let mut theta = DVector::zeros(l);
    for i in (0..l).rev() {
        let mut sum = 0.0;
        for k in (i + 1)..l {
            sum += a[(i, k)] * theta[k];
        }
        theta[i] = g[i] - sum;
    }
    theta
}
