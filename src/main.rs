mod narmax_method;
mod narmax_model;
mod regressor;
mod symbol;

use regressor::Regressor;
use std::collections::HashMap;
use std::fs;

use crate::narmax_method::NarmaxMethod;
use crate::narmax_method::frols::Frols;
use crate::narmax_model::NarmaxModel;

fn main() {
    const NY: usize = 2;
    const NU: usize = 2;
    const NE: usize = 2;
    const NON_LIN_LEN: usize = 3;

    let atoms = HashMap::from([("y", NY), ("u", NU)]);
    let regs = build_regressors(atoms, NON_LIN_LEN);
    let regs_str = regs
        .iter()
        .map(|r| format!("\t{}", r))
        .collect::<Vec<_>>()
        .join(",\n");

    println!("Candidatos:\n[\n{}\n]\n", regs_str);

    let datasets = [
        "res/ballbeam.csv",
        "res/wienerhammer.csv",
        "res/snls80.csv",
        "res/schroeder80.csv",
    ];

    for path in datasets {
        println!("====== {} ======", path);
        let (u, y) = load_csv(path);
        println!("Carregadas {} amostras", y.len());

        let ((u_tr, y_tr), (u_te, y_te)) = split_train_test(&u, &y, 0.3);

        println!("====== NARX ======");
        let model_narx = regs.clone().identify(Frols::new(0.001, 15), &y_tr, &u_tr);

        let y_hat_osa = predict_osa(&model_narx, &y_te, &u_te);
        let k_min = max_lag(&model_narx.regressors);
        let rmse_osa = rmse(&y_te[k_min..], &y_hat_osa);

        let y_hat_mpo = simulate_free_run(&model_narx, &y_te, &u_te);
        let rmse_mpo = rmse(&y_te[k_min..], &y_hat_mpo[k_min..]);

        println!("OSA RMSE: {:.6}", rmse_osa);
        println!("MPO RMSE: {:.6}", rmse_mpo);
        println!("gap MPO/OSA: {:.2}×\n", rmse_mpo / rmse_osa);

        println!("====== NARMAX ======");
        let mut model_narmax = model_narx;

        let regs_full = build_regressors(
            HashMap::from([("y", NY), ("u", NU), ("e", NE)]),
            NON_LIN_LEN,
        );

        for _ in 0..10 {
            let residue = compute_residual(&model_narmax, &y_tr, &u_tr);

            let new_model = regs_full.clone().identify_with_error(
                Frols::new(0.001, 15),
                &y_tr,
                &u_tr,
                &residue,
            );

            if converged(&new_model, &model_narmax, 1e-3) {
                println!("Converged with {} regressors", new_model.regressors.len());
                break;
            }

            model_narmax = new_model;
        }

        let y_hat_osa = predict_osa(&model_narmax, &y_te, &u_te);
        let k_min = max_lag(&model_narmax.regressors);
        let rmse_osa = rmse(&y_te[k_min..], &y_hat_osa);

        let y_hat_mpo = simulate_free_run(&model_narmax, &y_te, &u_te);
        let rmse_mpo = rmse(&y_te[k_min..], &y_hat_mpo[k_min..]);

        println!("OSA RMSE: {:.6}", rmse_osa);
        println!("MPO RMSE: {:.6}", rmse_mpo);
        println!("gap MPO/OSA: {:.2}×\n", rmse_mpo / rmse_osa);
    }
}

fn load_csv(path: &str) -> (Vec<f32>, Vec<f32>) {
    let data = fs::read_to_string(path).expect("falha ao ler CSV");
    let mut u = Vec::new();
    let mut y = Vec::new();
    for line in data.lines() {
        let mut cols = line.split(',');
        let _ = cols.next();
        u.push(cols.next().unwrap().parse::<f32>().unwrap());
        y.push(cols.next().unwrap().parse::<f32>().unwrap());
    }
    (u, y)
}

fn split_train_test(
    u: &[f32],
    y: &[f32],
    test_ratio: f32,
) -> ((Vec<f32>, Vec<f32>), (Vec<f32>, Vec<f32>)) {
    let total = y.len();
    let test_size = (total as f32 * test_ratio).round() as usize;
    let train_size = total - test_size;

    let u_train = u[..train_size].to_vec();
    let y_train = y[..train_size].to_vec();
    let u_test = u[train_size..].to_vec();
    let y_test = y[train_size..].to_vec();

    ((u_train, y_train), (u_test, y_test))
}

fn build_regressors(atoms: HashMap<&str, usize>, non_lin_len: usize) -> Vec<Regressor> {
    let mut base = vec![];

    for (name, max_lag) in atoms {
        for i in 1..=max_lag {
            base.push(Regressor::new(name, i));
        }
    }

    let mut all_terms = base.clone();
    let mut last_layer = base.clone();

    for _ in 2..=non_lin_len {
        let mut new_layer = vec![];

        for a in &base {
            for b in &last_layer {
                let prod = a * b;
                if !all_terms.contains(&prod) && !new_layer.contains(&prod) {
                    new_layer.push(prod);
                }
            }
        }

        all_terms.extend(new_layer.clone());
        last_layer = new_layer;
    }

    all_terms
}

fn predict_osa(model: &NarmaxModel, y: &[f32], u: &[f32]) -> Vec<f32> {
    let k_min = max_lag(&model.regressors);
    let n = y.len();
    let mut e = vec![0.0; n];
    let mut y_hat = Vec::with_capacity(n - k_min);

    for k in k_min..n {
        let samples = HashMap::from([("y", y), ("u", u), ("e", e.as_slice())]);
        let pred: f32 = model
            .regressors
            .iter()
            .zip(&model.theta)
            .map(|(r, &t)| t * r.eval_at(k, &samples).unwrap())
            .sum();
        e[k] = y[k] - pred;
        y_hat.push(pred);
    }

    y_hat
}

fn max_lag(regs: &[Regressor]) -> usize {
    regs.iter()
        .flat_map(|r| r.terms().iter().map(|s| s.index()))
        .max()
        .unwrap_or(0)
}

fn simulate_free_run(model: &NarmaxModel, y_init: &[f32], u: &[f32]) -> Vec<f32> {
    let k_min = max_lag(&model.regressors);
    let mut y_hat = y_init[..k_min].to_vec();
    let zeros = vec![0.0_f32; u.len()];

    for k in k_min..u.len() {
        let samples = HashMap::from([("y", y_hat.as_slice()), ("u", u), ("e", zeros.as_slice())]);
        let pred = model
            .regressors
            .iter()
            .zip(&model.theta)
            .map(|(r, &t)| t * r.eval_at(k, &samples).unwrap())
            .sum();

        y_hat.push(pred);
    }

    y_hat
}

fn rmse(y_true: &[f32], y_pred: &[f32]) -> f32 {
    let n = y_true.len() as f32;
    let sum_sq: f32 = y_true
        .iter()
        .zip(y_pred)
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    (sum_sq / n).sqrt()
}

fn compute_residual(model: &NarmaxModel, y: &[f32], u: &[f32]) -> Vec<f32> {
    let k_min = max_lag(&model.regressors);
    let n = y.len();

    let mut e = vec![0.0; n];

    for k in k_min..n {
        let samples = HashMap::from([("y", y), ("u", u), ("e", e.as_slice())]);
        let y_hat_k: f32 = model
            .regressors
            .iter()
            .zip(model.theta.iter())
            .map(|(r, t)| t * r.eval_at(k, &samples).unwrap())
            .sum();
        e[k] = y[k] - y_hat_k;
    }

    e
}

fn converged(new_model: &NarmaxModel, old_model: &NarmaxModel, theta_tol: f32) -> bool {
    let set_new = new_model
        .regressors
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>();
    let set_old = old_model
        .regressors
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>();

    if set_new != set_old {
        return false;
    }

    if new_model.theta.len() != old_model.theta.len() {
        return false;
    }

    let delta = new_model
        .theta
        .iter()
        .zip(old_model.theta.iter())
        .map(|(new_theta, old_theta)| new_theta - old_theta)
        .max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
        .unwrap_or(0.0);

    delta < theta_tol
}

trait NarmaxIdentify<M>
where
    M: NarmaxMethod,
{
    fn identify(self, method: M, y: &[f32], u: &[f32]) -> NarmaxModel;
    fn identify_with_error(self, method: M, y: &[f32], u: &[f32], e: &[f32]) -> NarmaxModel;
}

impl<M> NarmaxIdentify<M> for Vec<Regressor>
where
    M: NarmaxMethod,
{
    fn identify(self, method: M, y: &[f32], u: &[f32]) -> NarmaxModel {
        method.identify(self, y, u)
    }

    fn identify_with_error(self, method: M, y: &[f32], u: &[f32], e: &[f32]) -> NarmaxModel {
        method.identify_with_error(self, y, u, e)
    }
}
