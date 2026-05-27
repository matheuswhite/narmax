# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Coursework for a master's-level system identification course (`disciplinas/identificação`). Implements NARMAX (Nonlinear AutoRegressive Moving Average with eXogenous inputs) model identification in Rust, using **FROLS** (Forward Regression Orthogonal Least Squares) as the selection/estimation algorithm.

Linear algebra is `nalgebra` (`DMatrix`, `DVector`). Edition is Rust 2024 — recent toolchain required.

`README.md` at the project root contains the full experimental report (6 experiments comparing lag/degree/`l_max` configurations across 4 benchmark datasets, with out-of-sample validation results).

## Commands

```bash
cargo run --release     # builds release and runs identification + OSA/MPO validation on all 4 datasets
cargo build --release
cargo test              # all unit tests (~44 in symbol.rs and regressor.rs)
cargo test test_222     # run a single test by name
```

The "run" path always reads the four CSVs in `res/` (ballbeam, wienerhammer, snls80, schroeder80). The current configuration (lags, degree, `ρ`, `l_max`) lives directly in `main.rs` as the literals passed to `build_regressors(...)` and `Frols::new(...)`.

## Architecture

```
src/
├── symbol.rs                 atom: Symbol { name, index, power }
├── regressor.rs              Regressor (product of Symbols), Mul, eval, eval_at
├── narmax_model.rs           NarmaxModel struct (regressors, theta, phi, err, indices)
├── narmax_method/
│   ├── mod.rs                NarmaxMethod trait
│   └── frols.rs              FROLS implementation
└── main.rs                   build_regressors, dataset loading, validation pipeline
```

### Data flow

1. `build_regressors(ny, nu, d)` produces the candidate set: **all** monomials of total degree 1..=d over the atoms `[y(k-1)..y(k-ny), u(k-1)..u(k-nu)]`, deduplicated by order-insensitive equality.
2. `load_csv(path)` reads `(u, y)` from the `idx,u,y` CSVs in `res/`.
3. `split_train_test(u, y, ratio)` returns `((u_train, y_train), (u_test, y_test))` — **note the tuple ordering matches `(input, output)` per pair**.
4. `Vec<Regressor>::identify(method, y_train, u_train)` (via the `NarmaxIdentify` trait) dispatches to the chosen `NarmaxMethod`. Only `Frols` exists today.
5. FROLS returns a `NarmaxModel` with the selected regressors (in selection order) and their `theta` coefficients in the original basis.
6. `predict_osa(model, y_test, u_test)` produces one-step-ahead predictions; `simulate_free_run(model, y_test, u_test)` propagates the model autonomously using its own past predictions for `y` lags.

### Key conventions

- **`Symbol.index`** is the lag (`1` means `y(k-1)`). Index 0 means "current sample" but isn't emitted by `build_regressors`.
- **`Regressor::eval` and `eval_at`** take `&HashMap<&str, &[f32]>` keyed by variable name (`"y"`, `"u"`). All call sites must build this map before evaluating.
- **`Regressor::PartialEq`** is order-insensitive set-equality on `terms`, so `a*b == b*a`. This is how `build_regressors` dedups.
- **FROLS recovers `θ` in the original basis** by back-substitution on the upper-triangular `A` matrix. The Gram-Schmidt orthogonalization is implicit in the `α` coefficients stored during selection.
- **Stop criteria** in FROLS: `ESR = 1 − Σ err_i < ρ`, OR `l == l_max`, OR all remaining candidates are linearly dependent (`‖w_cand‖² < 1e-8`).
- **Tuple ordering trap**: `split_train_test` returns `((u_train, y_train), (u_test, y_test))` — first element of each pair is `u`. Destructure as `let ((u_tr, y_tr), (u_te, y_te)) = ...;` to keep names honest.

## Datasets in `res/`

Four CSV files with format `idx,u,y` (no header, comma-separated):

| File | Source | Samples |
|---|---|---:|
| `ballbeam.csv`     | DaISy ball-and-beam (`ballbeam.dat`)            | 1 000   |
| `wienerhammer.csv` | IFAC SYSID 2009 W-H benchmark (`WienerHammerBenchmark.csv`) | 188 000 |
| `snls80.csv`       | Silverbox SNLS 80mV (`SNLS80mV.csv`)            | 131 072 |
| `schroeder80.csv`  | Silverbox Schroeder 80mV (`Schroeder80mV.csv`)  | 131 072 |

Raw upstream files remain in `res/` and `res/SilverboxFiles/`, `res/WienerHammerstein2009Files/` — see the `.txt` / `README` documents alongside them for column semantics.

## Known limitations (not bugs)

- **No MA term identification** — despite the project name, only the NARX part is modeled. The MA part would require an outer loop that recomputes residuals and re-runs FROLS on an expanded candidate set with `e(k-i)` terms. The internal FROLS algorithm wouldn't change.
- **No regularization** — FROLS can pick terms with very large `|θ|` when there's near-colinearity in the candidate space. Watch for `|θ| ≫ |θ_lineares|` as a sign that the term is correcting numerical residual rather than carrying physical structure.
- **No constant/bias term** — the candidate set has no `1` term, so identified models assume zero-mean data. Datasets that need centering (e.g. Schroeder per its README) are loaded raw.
- **Wiener-Hammerstein is structurally inaccessible** to the current polynomial candidate set — see Experiment 6 in `README.md`. The selected model is always `y(k) ≈ 1.97 y(k-1) − 0.99 y(k-2)` regardless of lag/degree configuration.
