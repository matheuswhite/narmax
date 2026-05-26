# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Coursework for a master's-level system identification course (`disciplinas/identificação`). The goal is a Rust implementation of NARMAX (Nonlinear AutoRegressive Moving Average with eXogenous inputs) model identification. Linear algebra is provided by the `faer` crate (see `Cargo.toml`).

Edition is Rust 2024 — recent toolchain required.

## Commands

```bash
cargo run            # builds and runs the current main, which prints the generated regressor set
cargo build          # debug build
cargo build --release
cargo test           # run all tests
cargo test test_222  # run a single test by name
```

## Architecture

Three files in `src/`:

1. **`symbol.rs`** — `Symbol { name, index, power }`. The atom: one variable (e.g. `y`) at one lag, raised to one power. `pub(crate)` so `regressor.rs` can construct and pattern-match it.

2. **`regressor.rs`** — `Regressor { terms: Vec<Symbol> }`, a product of `Symbol`s. `Mul` for `&Regressor` combines two regressors by merging matching `(name, index)` terms and summing their powers — this is how higher-order regressor terms are constructed. `PartialEq` is order-insensitive (set-equality on `terms`), so `a*b == b*a`. All unit tests live here under `mod tests`.

3. **`main.rs`** — `fn build_regressors(y_len, u_len, non_lin_len) -> Vec<Regressor>` produces the full set of NARMAX regressor terms for `y_len` output lags, `u_len` input lags, and nonlinearity degree `non_lin_len`. It iteratively multiplies the previous degree's terms by the base `[y_1..y_n, u_1..u_n]` vector, deduplicating via the order-insensitive equality.

## Caveats for future work

- `build_regressors` only emits the degree-`non_lin_len` terms — it does not accumulate the lower-degree terms (linear, quadratic, …) that a full NARMAX regressor set would normally include.
- Identification of model coefficients (the "fit `faer` linear-system" step) is not yet implemented — only the symbolic term enumeration is.

## Data

`res/` contains benchmark datasets used in the course; not loaded by code yet:
- `ballbeam.dat` / `exchanger.dat` — DaISy-style two-column `(input, output)` text data, described in the matching `.txt` files.
- `SilverboxFiles/` and `WienerHammerstein2009Files/` — standard nonlinear-system-identification benchmarks (`.mat` + `.csv`).
