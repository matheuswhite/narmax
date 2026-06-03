//! Réplica fiel, em Rust, do notebook Python `atividade_03.ipynb`
//! (FROLS / MGS / SEMP por minimização do erro de simulação com poda).
//!
//! Reproduz o pipeline do notebook:
//!   - normalização z-score (estatísticas só do treino), split 60/40, subset de amostras;
//!   - dicionário polinomial COM termo constante (`combinations_with_replacement`);
//!   - FROLS/MGS para a estrutura inicial (via os métodos da biblioteca);
//!   - SEMP: parte do FROLS e faz `otimizar (L-BFGS) -> podar` minimizando a SOMA dos
//!     quadrados do erro de simulação livre (igual ao `scipy.optimize.minimize`).
//!
//! A otimização não-linear usa o crate `argmin` (L-BFGS + More-Thuente line search) com
//! gradiente por diferenças finitas (`finitediff`), em `f64` — como o scipy.

use std::fs;

use argmin::core::{CostFunction, Error, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use finitediff::vec::forward_diff;
use nalgebra::{DMatrix, DVector};

use narmax::narmax_method::NarmaxMethod;
use narmax::narmax_method::frols::Frols;
use narmax::narmax_method::mgs::Mgs;
use narmax::regressor::Regressor;
use narmax::symbol::Symbol;

const CLIP: f64 = 1e4;

struct DatasetCfg {
    name: &'static str,
    path: &'static str,
    n_use: Option<usize>,
    ny: usize,
    nu: usize,
    degree: usize,
    max_terms: usize,
    rho: f32, // = 1 - err_threshold do notebook
}

fn main() {
    let datasets = [
        DatasetCfg {
            name: "Ball & Beam",
            path: "res/ballbeam.csv",
            n_use: None,
            ny: 2,
            nu: 2,
            degree: 2,
            max_terms: 12,
            rho: 0.001, // err_thr = 0.999
        },
        DatasetCfg {
            name: "Silverbox (SNLS80)",
            path: "res/snls80.csv",
            n_use: Some(8000),
            ny: 2,
            nu: 2,
            degree: 3,
            max_terms: 15,
            rho: 0.0001, // err_thr = 0.9999
        },
        DatasetCfg {
            name: "Wiener-Hammerstein",
            path: "res/wienerhammer.csv",
            n_use: Some(8000),
            ny: 3,
            nu: 3,
            degree: 3,
            max_terms: 15,
            rho: 0.0001, // err_thr = 0.9999
        },
    ];

    for cfg in &datasets {
        compare_dataset(cfg);
    }
}

fn compare_dataset(cfg: &DatasetCfg) {
    let (u_raw, y_raw) = load_csv(cfg.path);
    let (u_tr, y_tr, u_val, y_val) = split_normalize(&u_raw, &y_raw, cfg.n_use, 0.6);

    let max_lag = cfg.ny.max(cfg.nu);
    let (candidates, monomials) = gen_candidates(cfg.ny, cfg.nu, cfg.degree);

    println!("\n{}", "═".repeat(64));
    println!(
        "  Dataset: {}  |  ny={}, nu={}, grau={}",
        cfg.name, cfg.ny, cfg.nu, cfg.degree
    );
    println!(
        "  Treino: {} amostras | Validação: {} amostras | Candidatos: {}",
        u_tr.len(),
        u_val.len(),
        candidates.len()
    );
    println!("{}", "═".repeat(64));

    // ── FROLS ──────────────────────────────────────────────────────────────
    let sel_f = select(
        Frols::new(cfg.rho, cfg.max_terms),
        &candidates,
        &y_tr,
        &u_tr,
    );
    let monos_f: Vec<Vec<usize>> = sel_f.iter().map(|&i| monomials[i].clone()).collect();
    let theta_f = ls_estimate(&monos_f, &y_tr, &u_tr, cfg.ny, max_lag);
    report("FROLS", &monos_f, &theta_f, cfg, &y_tr, &u_tr, &y_val, &u_val, None);

    // ── MGS ────────────────────────────────────────────────────────────────
    let sel_m = select(Mgs::new(cfg.rho, cfg.max_terms), &candidates, &y_tr, &u_tr);
    let monos_m: Vec<Vec<usize>> = sel_m.iter().map(|&i| monomials[i].clone()).collect();
    let theta_m = ls_estimate(&monos_m, &y_tr, &u_tr, cfg.ny, max_lag);
    report("MGS", &monos_m, &theta_m, cfg, &y_tr, &u_tr, &y_val, &u_val, None);

    // ── SEMP (parte do FROLS, otimiza erro de simulação + poda) ─────────────
    let t0 = std::time::Instant::now();
    let (monos_s, theta_s) = semp(&monos_f, &theta_f, &y_tr, &u_tr, cfg.ny, cfg.nu, 0.05, max_lag);
    let dt = t0.elapsed();
    report(
        "SEMP",
        &monos_s,
        &theta_s,
        cfg,
        &y_tr,
        &u_tr,
        &y_val,
        &u_val,
        Some(dt),
    );
}

/// Roda um método de seleção (FROLS/MGS) da biblioteca e devolve os índices
/// selecionados no conjunto de candidatos.
fn select<M: NarmaxMethod>(method: M, candidates: &[Regressor], y: &[f64], u: &[f64]) -> Vec<usize> {
    let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
    let u32: Vec<f32> = u.iter().map(|&v| v as f32).collect();
    let model = method.identify(candidates.to_vec(), &y32, &u32);
    model.selected_indices.unwrap_or_default()
}

fn report(
    label: &str,
    monos: &[Vec<usize>],
    theta: &[f64],
    cfg: &DatasetCfg,
    y_tr: &[f64],
    u_tr: &[f64],
    y_val: &[f64],
    u_val: &[f64],
    elapsed: Option<std::time::Duration>,
) {
    let max_lag = cfg.ny.max(cfg.nu);

    let ys_tr = simulate_narx(y_tr, u_tr, theta, monos, cfg.ny, cfg.nu);
    let ys_val = simulate_narx(y_val, u_val, theta, monos, cfg.ny, cfg.nu);

    let (rmse_tr, nrmse_tr, r2_tr) = metrics(&y_tr[max_lag..], &ys_tr[max_lag..]);
    let (rmse_val, nrmse_val, r2_val) = metrics(&y_val[max_lag..], &ys_val[max_lag..]);

    let time_str = match elapsed {
        Some(d) => format!(", {:.2?}", d),
        None => String::new(),
    };
    println!("\n  ── {label}  ({} termos{})", monos.len(), time_str);
    println!("     Termos: {}", labels(monos, cfg.ny, cfg.nu));
    println!(
        "     [treino]    RMSE={:.5}  NRMSE={:.4}  R²={:.4}",
        rmse_tr, nrmse_tr, r2_tr
    );
    println!(
        "     [validação] RMSE={:.5}  NRMSE={:.4}  R²={:.4}",
        rmse_val, nrmse_val, r2_val
    );
}

// ─────────────────────────── Dados / normalização ──────────────────────────

fn load_csv(path: &str) -> (Vec<f64>, Vec<f64>) {
    let data = fs::read_to_string(path).expect("falha ao ler CSV");
    let mut u = Vec::new();
    let mut y = Vec::new();
    for line in data.lines() {
        let mut cols = line.split(',');
        let _ = cols.next();
        u.push(cols.next().unwrap().parse::<f64>().unwrap());
        y.push(cols.next().unwrap().parse::<f64>().unwrap());
    }
    (u, y)
}

/// z-score com estatísticas só do treino + split treino/validação (igual ao notebook).
fn split_normalize(
    u_full: &[f64],
    y_full: &[f64],
    n_use: Option<usize>,
    train_frac: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = n_use.unwrap_or(u_full.len()).min(u_full.len());
    let u = &u_full[..n];
    let y = &y_full[..n];
    let n_train = (n as f64 * train_frac) as usize;

    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let std = |s: &[f64], m: f64| {
        (s.iter().map(|v| (v - m).powi(2)).sum::<f64>() / s.len() as f64).sqrt() + 1e-12
    };

    let (um, ym) = (mean(&u[..n_train]), mean(&y[..n_train]));
    let (us, ys) = (std(&u[..n_train], um), std(&y[..n_train], ym));

    let un: Vec<f64> = u.iter().map(|v| (v - um) / us).collect();
    let yn: Vec<f64> = y.iter().map(|v| (v - ym) / ys).collect();

    (
        un[..n_train].to_vec(),
        yn[..n_train].to_vec(),
        un[n_train..].to_vec(),
        yn[n_train..].to_vec(),
    )
}

// ─────────────────────────── Candidatos polinomiais ────────────────────────

/// Gera os monômios (tuplas de índices de variáveis de atraso, COM repetição) de grau
/// 0..=degree, incluindo o termo constante `()`. Espelha `gen_monomials` do notebook.
fn gen_candidates(ny: usize, nu: usize, degree: usize) -> (Vec<Regressor>, Vec<Vec<usize>>) {
    let n_vars = ny + nu;
    let mut monos: Vec<Vec<usize>> = vec![vec![]]; // grau 0 = constante
    for d in 1..=degree {
        let mut cur = Vec::new();
        cwr(n_vars, d, 0, &mut cur, &mut monos);
    }
    let regs = monos.iter().map(|m| reg_from_mono(m, ny, nu)).collect();
    (regs, monos)
}

/// combinations_with_replacement(range(n), k): índices não-decrescentes.
fn cwr(n: usize, k: usize, start: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    if cur.len() == k {
        out.push(cur.clone());
        return;
    }
    for i in start..n {
        cur.push(i);
        cwr(n, k, i, cur, out);
        cur.pop();
    }
}

/// Índice de variável de atraso -> regressor atômico. 0..ny = y(k-i), ny.. = u(k-i).
fn atom(idx: usize, ny: usize, _nu: usize) -> Regressor {
    if idx < ny {
        Regressor::new("y", idx + 1)
    } else {
        Regressor::new("u", idx - ny + 1)
    }
}

fn reg_from_mono(mono: &[usize], ny: usize, nu: usize) -> Regressor {
    if mono.is_empty() {
        return Regressor::from(Vec::<Symbol>::new()); // termo constante = 1
    }
    let mut r = atom(mono[0], ny, nu);
    for &idx in &mono[1..] {
        r = &r * &atom(idx, ny, nu);
    }
    r
}

fn labels(monos: &[Vec<usize>], ny: usize, nu: usize) -> String {
    monos
        .iter()
        .map(|m| format!("{}", reg_from_mono(m, ny, nu)))
        .collect::<Vec<_>>()
        .join(", ")
}

// ─────────────────────────── Simulação / métricas ──────────────────────────

/// Simulação livre (multi-passo), igual a `simulate_narx` do notebook: semeia os
/// `max_lag` primeiros valores com `y_seed`, propaga as próprias predições e satura
/// em ±CLIP para conter divergência numérica.
fn simulate_narx(
    y_seed: &[f64],
    u: &[f64],
    theta: &[f64],
    monos: &[Vec<usize>],
    ny: usize,
    nu: usize,
) -> Vec<f64> {
    let max_lag = ny.max(nu);
    let n = u.len();
    let mut y = vec![0.0_f64; n];
    y[..max_lag].copy_from_slice(&y_seed[..max_lag]);

    for k in max_lag..n {
        // lag_vars = [y(k-1)..y(k-ny), u(k-1)..u(k-nu)]
        let mut pred = 0.0;
        for (t, mono) in monos.iter().enumerate() {
            let mut v = 1.0;
            for &idx in mono {
                v *= if idx < ny {
                    y[k - (idx + 1)]
                } else {
                    u[k - (idx - ny + 1)]
                };
            }
            pred += theta[t] * v;
        }
        y[k] = pred.clamp(-CLIP, CLIP);
    }
    y
}

/// SOMA dos quadrados do erro de simulação livre no treino (objetivo do SEMP).
fn sim_sse(theta: &[f64], monos: &[Vec<usize>], y: &[f64], u: &[f64], ny: usize, nu: usize) -> f64 {
    let max_lag = ny.max(nu);
    let y_sim = simulate_narx(y, u, theta, monos, ny, nu);
    let mut sse = 0.0;
    for k in max_lag..y.len() {
        let e = y_sim[k] - y[k];
        if !e.is_finite() {
            return 1e12;
        }
        sse += e * e;
    }
    sse
}

fn metrics(y_true: &[f64], y_pred: &[f64]) -> (f64, f64, f64) {
    let n = y_true.len() as f64;
    let mean = y_true.iter().sum::<f64>() / n;
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    let ss_tot: f64 = y_true.iter().map(|a| (a - mean).powi(2)).sum();
    let rmse = (ss_res / n).sqrt();
    let std = (ss_tot / n).sqrt() + 1e-12;
    let nrmse = rmse / std;
    let r2 = 1.0 - ss_res / (ss_tot + 1e-12);
    (rmse, nrmse, r2)
}

// ─────────────────────────── LS de uma passada (theta inicial) ─────────────

/// Estimativa OLS (predição de 1 passo) nos monômios selecionados — corresponde ao
/// `np.linalg.lstsq` que o notebook usa para a estrutura do FROLS/MGS.
fn ls_estimate(monos: &[Vec<usize>], y: &[f64], u: &[f64], ny: usize, max_lag: usize) -> Vec<f64> {
    let p = monos.len();
    if p == 0 {
        return vec![];
    }
    let n_eff = y.len() - max_lag;
    let psi = DMatrix::from_fn(n_eff, p, |i, j| {
        let k = max_lag + i;
        let mut v = 1.0;
        for &idx in &monos[j] {
            v *= if idx < ny {
                y[k - (idx + 1)]
            } else {
                u[k - (idx - ny + 1)]
            };
        }
        v
    });
    let yv = DVector::from_fn(n_eff, |i, _| y[max_lag + i]);

    let xtx = psi.transpose() * &psi;
    let xty = psi.transpose() * yv;
    xtx.lu()
        .solve(&xty)
        .map(|t| t.iter().copied().collect())
        .unwrap_or_else(|| vec![0.0; p])
}

// ─────────────────────────── SEMP (otimizar + podar) ───────────────────────

/// Problema de otimização para o L-BFGS: minimizar a SSE de simulação livre sobre theta,
/// com a estrutura (`monos`) fixa.
struct SimProblem {
    monos: Vec<Vec<usize>>,
    y: Vec<f64>,
    u: Vec<f64>,
    ny: usize,
    nu: usize,
}

impl SimProblem {
    fn loss(&self, theta: &[f64]) -> f64 {
        sim_sse(theta, &self.monos, &self.y, &self.u, self.ny, self.nu)
    }
}

impl CostFunction for SimProblem {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, p: &Self::Param) -> Result<f64, Error> {
        Ok(self.loss(p))
    }
}

impl Gradient for SimProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        let f = |x: &Vec<f64>| -> Result<f64, Error> { Ok(self.loss(x)) };
        forward_diff(&f)(p)
    }
}

/// Minimiza a SSE de simulação livre via L-BFGS (More-Thuente line search), partindo de
/// `theta0`. Em caso de falha do otimizador (ex.: line search), devolve `theta0`.
fn optimize(
    monos: &[Vec<usize>],
    theta0: &[f64],
    y: &[f64],
    u: &[f64],
    ny: usize,
    nu: usize,
    max_iters: u64,
) -> (Vec<f64>, f64) {
    let problem = SimProblem {
        monos: monos.to_vec(),
        y: y.to_vec(),
        u: u.to_vec(),
        ny,
        nu,
    };
    let fallback = problem.loss(theta0);

    let linesearch = match MoreThuenteLineSearch::new().with_c(1e-4, 0.9) {
        Ok(ls) => ls,
        Err(_) => return (theta0.to_vec(), fallback),
    };
    let solver = LBFGS::new(linesearch, 7);

    let result = Executor::new(problem, solver)
        .configure(|s| s.param(theta0.to_vec()).max_iters(max_iters))
        .run();

    match result {
        Ok(res) => {
            let st = res.state();
            match st.get_best_param() {
                Some(p) => (p.clone(), st.get_best_cost()),
                None => (theta0.to_vec(), fallback),
            }
        }
        Err(_) => (theta0.to_vec(), fallback),
    }
}

/// SEMP: a partir da estrutura inicial (FROLS), repete `otimizar -> podar`. Remove o
/// termo cuja remoção menos aumenta a SSE de simulação, enquanto esse aumento relativo
/// for ≤ `prune_tol`. Espelha a função `semp` do notebook.
#[allow(clippy::too_many_arguments)]
fn semp(
    initial_monos: &[Vec<usize>],
    initial_theta: &[f64],
    y: &[f64],
    u: &[f64],
    ny: usize,
    nu: usize,
    prune_tol: f64,
    _max_lag: usize,
) -> (Vec<Vec<usize>>, Vec<f64>) {
    let mut monos: Vec<Vec<usize>> = initial_monos.to_vec();
    let mut theta: Vec<f64> = initial_theta.to_vec();

    loop {
        // Fase 1: otimização do erro de simulação.
        let (opt_theta, baseline) = optimize(&monos, &theta, y, u, ny, nu, 300);
        theta = opt_theta;

        if monos.len() <= 1 {
            break;
        }

        // Fase 2: poda — encontra o termo de menor aumento relativo ao removê-lo.
        let mut best_i = usize::MAX;
        let mut best_rel = f64::INFINITY;
        let mut best_theta = Vec::new();
        let mut best_monos = Vec::new();

        for i in 0..monos.len() {
            let mut test_monos = monos.clone();
            test_monos.remove(i);
            let mut test_theta = theta.clone();
            test_theta.remove(i);

            let (pt, loss) = optimize(&test_monos, &test_theta, y, u, ny, nu, 100);
            let rel_inc = (loss - baseline) / (baseline.abs() + 1e-10);

            if rel_inc < best_rel {
                best_rel = rel_inc;
                best_i = i;
                best_theta = pt;
                best_monos = test_monos;
            }
        }

        if best_rel <= prune_tol && best_i != usize::MAX {
            monos = best_monos;
            theta = best_theta;
        } else {
            break;
        }
    }

    // Otimização final.
    let (final_theta, _) = optimize(&monos, &theta, y, u, ny, nu, 600);
    (monos, final_theta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_plus_two_lags_candidate_count() {
        // ny=2, nu=2, degree=2 -> 1 (const) + 4 + 10 = 15, como no notebook.
        let (regs, monos) = gen_candidates(2, 2, 2);
        assert_eq!(regs.len(), 15);
        assert_eq!(monos.len(), 15);
        assert!(monos[0].is_empty(), "primeiro candidato deve ser a constante");
    }

    #[test]
    fn simulate_matches_manual_linear() {
        // y(k) = 0.5 y(k-1) + 0.3 u(k-1); monômios: y(k-1) e u(k-1).
        let monos = vec![vec![0_usize], vec![2_usize]]; // ny=2,nu=2 -> idx2 = u(k-1)
        let theta = vec![0.5, 0.3];
        let y_seed = vec![1.0, 0.0];
        let u = vec![0.0, 1.0, 1.0, 1.0];
        let ys = simulate_narx(&y_seed, &u, &theta, &monos, 2, 2);
        // k=2: 0.5*y[1]+0.3*u[1] = 0.5*0 + 0.3*1 = 0.3
        assert!((ys[2] - 0.3).abs() < 1e-12);
        // k=3: 0.5*y[2]+0.3*u[2] = 0.5*0.3 + 0.3 = 0.45
        assert!((ys[3] - 0.45).abs() < 1e-12);
    }
}
