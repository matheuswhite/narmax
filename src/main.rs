mod regressor;
mod symbol;

use nalgebra::{DMatrix, DVector};
use regressor::Regressor;
use std::collections::HashMap;
use std::fs;

fn main() {
    let regs = build_regressors(3, 3, 3);
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

        let model = regs.clone().identify(Frols::new(0.001, 15), &y, &u);

        println!("Selecionados ({} regressores):", model.regressors.len());
        for (i, r) in model.regressors.iter().enumerate() {
            let theta = model.theta[i];
            let err = model.err.as_ref().map(|v| v[i]).unwrap_or(0.0);
            println!("  θ[{}] = {:>+12.6}  ERR = {:.6}   {}", i, theta, err, r);
        }
        println!();
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

fn build_regressors(y_len: usize, u_len: usize, non_lin_len: usize) -> Vec<Regressor> {
    let y = (1..=y_len)
        .map(|i| Regressor::new("y", i))
        .collect::<Vec<_>>();
    let u = (1..=u_len)
        .map(|i| Regressor::new("u", i))
        .collect::<Vec<_>>();

    let yu_lhs = [y, u].concat();
    let mut all_terms = yu_lhs.clone();
    let mut last_layer = yu_lhs.clone();

    for _ in 2..=non_lin_len {
        let mut new_layer = vec![];

        for a in &yu_lhs {
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

#[derive(Debug)]
struct NarmaxModel {
    regressors: Vec<Regressor>,
    theta: Vec<f32>,
    phi: Option<DMatrix<f32>>,
    err: Option<Vec<f32>>,
    selected_indices: Option<Vec<usize>>,
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

trait NarmaxMethod {
    fn identify(self, regressors: Vec<Regressor>, y: &[f32], u: &[f32]) -> NarmaxModel;
}

#[derive(Clone, Copy)]
struct Frols {
    rho: f32,
    l_max: usize,
}

impl Frols {
    fn new(rho: f32, l_max: usize) -> Self {
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
}

impl NarmaxMethod for Frols {
    fn identify(self, regressors: Vec<Regressor>, y: &[f32], u: &[f32]) -> NarmaxModel {
        /* Setup */
        let phi = Self::build_initial_phi(&regressors, y, u);
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
