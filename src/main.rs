use std::collections::HashMap;
use std::fs;
use std::time::Instant;

use narmax::narmax_method::NarmaxMethod;
use narmax::narmax_method::frols::Frols;
use narmax::narmax_method::mgs::Mgs;
use narmax::narmax_method::semp::Semp;
use narmax::narmax_model::{NarmaxModel, max_lag, rmse};
use narmax::refine::refine_simulation_error;
use narmax::regressor::{Regressor, build_regressors};

fn main() {
    const NY: usize = 2;
    const NU: usize = 2;
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
        run_narx(
            "FROLS",
            Frols::new(0.001, 15),
            &regs,
            &y_tr,
            &u_tr,
            &y_te,
            &u_te,
        );
        run_narx(
            "MGS",
            Mgs::new(0.001, 15),
            &regs,
            &y_tr,
            &u_tr,
            &y_te,
            &u_te,
        );
        run_narx(
            "SEMP",
            Semp::new(15, 1e-3, 5),
            &regs,
            &y_tr,
            &u_tr,
            &y_te,
            &u_te,
        );

        // Extra: refino dos coeficientes por minimização do erro de simulação livre
        // (otimização não-linear estilo `scipy.optimize.minimize` do notebook Python).
        // Parte do modelo FROLS, como no notebook.
        println!("====== NARX + refino por erro de simulação (estilo notebook) ======");
        run_narx_refined(
            "FROLS",
            Frols::new(0.001, 15),
            &regs,
            &y_tr,
            &u_tr,
            &y_te,
            &u_te,
        );
    }
}

/// Identify a NARX model with `method`, then print OSA/MPO RMSE on the test set.
fn run_narx<M: NarmaxMethod>(
    label: &str,
    method: M,
    regs: &[Regressor],
    y_tr: &[f32],
    u_tr: &[f32],
    y_te: &[f32],
    u_te: &[f32],
) {
    let model = regs.to_vec().identify(method, y_tr, u_tr);
    let k_min = max_lag(&model.regressors);
    let rmse_osa = rmse(&y_te[k_min..], &model.predict_osa(y_te, u_te));
    let y_mpo = model.simulate_free_run(y_te, u_te);
    let rmse_mpo = rmse(&y_te[k_min..], &y_mpo[k_min..]);

    println!("--- {label} ({} regs) ---", model.regressors.len());
    println!("OSA RMSE: {:.6}", rmse_osa);
    println!("MPO RMSE: {:.6}", rmse_mpo);
    println!("gap MPO/OSA: {:.2}×\n", rmse_mpo / rmse_osa);
}

/// Identify a NARX model, then refine its coefficients by minimizing the free-run
/// simulation error (nonlinear Nelder-Mead). Prints the LS baseline and the refined
/// result side by side, plus the wall-clock cost of the refinement.
fn run_narx_refined<M: NarmaxMethod>(
    label: &str,
    method: M,
    regs: &[Regressor],
    y_tr: &[f32],
    u_tr: &[f32],
    y_te: &[f32],
    u_te: &[f32],
) {
    const REFINE_ITERS: usize = 300;

    let model = regs.to_vec().identify(method, y_tr, u_tr);
    let k_min = max_lag(&model.regressors);

    let osa_ls = rmse(&y_te[k_min..], &model.predict_osa(y_te, u_te));
    let mpo_ls = model.simulate_free_run(y_te, u_te);
    let mpo_ls = rmse(&y_te[k_min..], &mpo_ls[k_min..]);

    let t0 = Instant::now();
    let theta_ref = refine_simulation_error(&model.regressors, &model.theta, y_tr, u_tr, REFINE_ITERS);
    let dt = t0.elapsed();

    let refined = NarmaxModel {
        regressors: model.regressors.clone(),
        theta: theta_ref,
        phi: None,
        err: None,
        selected_indices: None,
    };
    let osa_ref = rmse(&y_te[k_min..], &refined.predict_osa(y_te, u_te));
    let mpo_ref = refined.simulate_free_run(y_te, u_te);
    let mpo_ref = rmse(&y_te[k_min..], &mpo_ref[k_min..]);

    println!("--- {label} ({} regs) ---", model.regressors.len());
    println!("LS         -> OSA {:.6} | MPO {:.6}", osa_ls, mpo_ls);
    println!(
        "sim. error -> OSA {:.6} | MPO {:.6}  (refino: {} iters em {:.2?})",
        osa_ref, mpo_ref, REFINE_ITERS, dt
    );
    println!();
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

trait NarmaxIdentify<M>
where
    M: NarmaxMethod,
{
    fn identify(self, method: M, y: &[f32], u: &[f32]) -> NarmaxModel;
}

impl<M> NarmaxIdentify<M> for Vec<Regressor>
where
    M: NarmaxMethod,
{
    fn identify(self, method: M, y: &[f32], u: &[f32]) -> NarmaxModel {
        method.identify(self, y, u)
    }
}
