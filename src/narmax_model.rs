use crate::regressor::Regressor;
use nalgebra::DMatrix;
use std::collections::HashMap;

#[derive(Debug)]
pub struct NarmaxModel {
    pub regressors: Vec<Regressor>,
    pub theta: Vec<f32>,
    pub phi: Option<DMatrix<f32>>,
    pub err: Option<Vec<f32>>,
    pub selected_indices: Option<Vec<usize>>,
}

impl NarmaxModel {
    /// Max lag over the SELECTED regressors (simulation/prediction seed offset).
    pub fn max_lag(&self) -> usize {
        max_lag(&self.regressors)
    }

    /// One-step-ahead prediction: uses the true past `y`/`u` for the lags.
    pub fn predict_osa(&self, y: &[f32], u: &[f32]) -> Vec<f32> {
        let k_min = self.max_lag();
        let n = y.len();
        let mut y_hat = Vec::with_capacity(n - k_min);

        for k in k_min..n {
            let samples = HashMap::from([("y", y), ("u", u)]);
            let pred: f32 = self
                .regressors
                .iter()
                .zip(&self.theta)
                .map(|(r, &t)| t * r.eval_at(k, &samples).unwrap())
                .sum();
            y_hat.push(pred);
        }

        y_hat
    }

    /// Free-run (model-predicted-output) simulation: seeds with `y_init[..k_min]`,
    /// then propagates the model autonomously using its own predictions for `y`
    /// lags and the true `u`.
    pub fn simulate_free_run(&self, y_init: &[f32], u: &[f32]) -> Vec<f32> {
        let k_min = self.max_lag();
        let mut y_hat = y_init[..k_min].to_vec();

        for k in k_min..u.len() {
            let samples = HashMap::from([("y", y_hat.as_slice()), ("u", u)]);
            let pred = self
                .regressors
                .iter()
                .zip(&self.theta)
                .map(|(r, &t)| t * r.eval_at(k, &samples).unwrap())
                .sum();

            y_hat.push(pred);
        }

        y_hat
    }
}

/// Max lag over a slice of regressors' symbols.
pub fn max_lag(regs: &[Regressor]) -> usize {
    regs.iter()
        .flat_map(|r| r.terms().iter().map(|s| s.index()))
        .max()
        .unwrap_or(0)
}

pub fn rmse(y_true: &[f32], y_pred: &[f32]) -> f32 {
    (mse(y_true, y_pred)).sqrt()
}

/// Mean squared error. With a free-run output this is the MSSE used by SEMP.
pub fn mse(y_true: &[f32], y_pred: &[f32]) -> f32 {
    let n = y_true.len() as f32;
    let sum_sq: f32 = y_true
        .iter()
        .zip(y_pred)
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    sum_sq / n
}
