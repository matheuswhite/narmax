use std::{collections::HashMap, fmt::Display, ops::Mul};

use crate::symbol::Symbol;

#[derive(Clone, Debug, Eq, Hash)]
pub struct Regressor {
    terms: Vec<Symbol>,
}

impl Regressor {
    pub fn new(name: impl AsRef<str>, index: usize) -> Self {
        Self {
            terms: vec![Symbol::new(name, index, 1)],
        }
    }

    pub fn terms(&self) -> &[Symbol] {
        &self.terms
    }

    pub fn eval(&self, samples: &HashMap<&str, &[f32]>) -> Vec<f32> {
        let terms_samples: Vec<_> = self
            .terms
            .iter()
            .map(|term| term.eval(samples[term.name()]))
            .collect();
        let Some(least_len) = terms_samples.iter().map(|term| term.len()).min() else {
            return vec![];
        };

        let terms_samples: Vec<_> = terms_samples
            .into_iter()
            .map(|term| {
                let index = term.len() - least_len;
                term.into_iter().skip(index).collect::<Vec<_>>()
            })
            .collect();

        vec![0.0; least_len]
            .into_iter()
            .enumerate()
            .map(|(i, _)| terms_samples.iter().fold(1.0, |acc, el| acc * el[i]))
            .collect()
    }

    pub fn eval_at(&self, k: usize, samples: &HashMap<&str, &[f32]>) -> Option<f32> {
        self.terms().iter().try_fold(1.0, |acc, term| {
            let s = samples.get(term.name())?;
            let i = k.checked_sub(term.index())?;
            let v = s.get(i)?;

            Some(acc * v.powi(term.power() as i32))
        })
    }
}

impl From<Vec<Symbol>> for Regressor {
    fn from(mut terms: Vec<Symbol>) -> Self {
        terms.sort();
        Self { terms }
    }
}

impl PartialEq for Regressor {
    fn eq(&self, other: &Self) -> bool {
        if self.terms.len() != other.terms.len() {
            return false;
        }
        self.terms.iter().all(|t| other.terms.contains(t))
    }
}

impl Display for Regressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let terms_with_powers: Vec<String> = self
            .terms
            .iter()
            .map(|term| {
                let index = if term.index() == 0 {
                    "".to_string()
                } else {
                    format!("-{}", term.index())
                };
                let var = format!("{}(k{})", term.name(), index);
                if term.power() == 1 {
                    var
                } else {
                    format!("{}^{}", var, term.power())
                }
            })
            .collect();

        write!(f, "{}", terms_with_powers.join("*"))
    }
}

impl Mul for &Regressor {
    type Output = Regressor;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut combined_terms = self.terms.clone();

        for rhs_term in &rhs.terms {
            if let Some(pos) = combined_terms
                .iter()
                .position(|t| t.name() == rhs_term.name() && t.index() == rhs_term.index())
            {
                let new_power = combined_terms[pos].power() + rhs_term.power();
                combined_terms[pos].set_power(new_power);
            } else {
                combined_terms.push(rhs_term.clone());
            }
        }

        Regressor::from(combined_terms)
    }
}

impl Mul for Regressor {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_regressors;

    fn assert_same_regressor_set(left: &[Regressor], right: &[Regressor]) {
        assert_eq!(
            left.len(),
            right.len(),
            "len mismatch: {} vs {}",
            left.len(),
            right.len()
        );
        for r in left {
            assert!(right.contains(r), "missing in right: {}", r);
        }
        for r in right {
            assert!(left.contains(r), "missing in left: {}", r);
        }
    }

    #[test]
    fn test_regressor_display() {
        let symbol = Regressor::new("x", 0);
        assert_eq!(symbol.to_string(), "x(k)");
        let symbol = Regressor::from(vec![Symbol::new("x", 0, 2), Symbol::new("y", 0, 3)]);
        assert_eq!(symbol.to_string(), "x(k)^2*y(k)^3");
    }

    #[test]
    fn test_regressor_mul() {
        let symbol1 = Regressor::new("x", 0);
        let symbol2 = Regressor::new("x", 0);
        let result = symbol1 * symbol2;
        assert_eq!(result.to_string(), "x(k)^2");
        let symbol1 = Regressor::new("x", 0);
        let symbol2 = Regressor::new("y", 0);
        let result = symbol1 * symbol2;
        assert_eq!(result.to_string(), "x(k)*y(k)");
        let symbol1 = Regressor::from(vec![Symbol::new("x", 0, 2), Symbol::new("y", 0, 3)]);
        let symbol2 = Regressor::from(vec![Symbol::new("y", 0, 1), Symbol::new("z", 0, 4)]);
        let result = symbol1 * symbol2;
        assert_eq!(result.to_string(), "x(k)^2*y(k)^4*z(k)^4");
    }

    #[test]
    fn test_regressor_eq() {
        let a = Regressor::new("a", 0);
        let b = Regressor::new("b", 0);

        assert_eq!(&a * &b, &b * &a);
    }

    #[test]
    fn test_222() {
        let output = build_regressors(HashMap::from([("y", 2), ("u", 2)]), 2);
        let a = Regressor::new("y", 1);
        let b = Regressor::new("y", 2);
        let c = Regressor::new("u", 1);
        let d = Regressor::new("u", 2);

        let expected = vec![
            a.clone(),
            b.clone(),
            c.clone(),
            d.clone(),
            &a * &a,
            &a * &b,
            &a * &c,
            &a * &d,
            &b * &b,
            &b * &c,
            &b * &d,
            &c * &c,
            &c * &d,
            &d * &d,
        ];

        assert_same_regressor_set(&output, &expected);
    }

    #[test]
    fn from_vec_symbol_preserves_terms() {
        let terms = vec![Symbol::new("x", 0, 2), Symbol::new("y", 1, 1)];
        let r = Regressor::from(terms.clone());
        assert_eq!(r.terms(), terms.as_slice());
    }

    #[test]
    fn new_creates_single_atom_with_power_one() {
        let r = Regressor::new("y", 2);
        assert_eq!(r.terms(), &[Symbol::new("y", 2, 1)]);
    }

    #[test]
    fn display_with_zero_index_omits_lag() {
        let r = Regressor::new("x", 0);
        assert_eq!(r.to_string(), "x(k)");
    }

    #[test]
    fn display_with_power_one_omits_exponent() {
        let r = Regressor::new("y", 1);
        assert_eq!(r.to_string(), "y(k-1)");
    }

    #[test]
    fn mul_is_commutative() {
        let a = Regressor::new("y", 1);
        let b = Regressor::new("u", 2);
        assert_eq!(&a * &b, &b * &a);
    }

    #[test]
    fn mul_is_associative() {
        let a = Regressor::new("y", 1);
        let b = Regressor::new("y", 2);
        let c = Regressor::new("u", 1);
        assert_eq!(&(&a * &b) * &c, &a * &(&b * &c));
    }

    #[test]
    fn mul_merges_repeated_atoms() {
        let a = Regressor::new("y", 1);
        let prod = &a * &a;
        assert_eq!(prod.terms(), &[Symbol::new("y", 1, 2)]);
        let cubed = &prod * &a;
        assert_eq!(cubed.terms(), &[Symbol::new("y", 1, 3)]);
    }

    #[test]
    fn mul_owned_matches_borrowed() {
        let a = Regressor::new("y", 1);
        let b = Regressor::new("u", 2);
        assert_eq!(a.clone() * b.clone(), &a * &b);
    }

    #[test]
    fn eq_is_symmetric() {
        let a = Regressor::new("y", 1);
        let b = Regressor::new("y", 1);
        assert_eq!(a == b, b == a);
        assert!(a == b);
    }

    #[test]
    fn eq_is_order_insensitive() {
        let yu = Regressor::from(vec![Symbol::new("y", 1, 1), Symbol::new("u", 1, 1)]);
        let uy = Regressor::from(vec![Symbol::new("u", 1, 1), Symbol::new("y", 1, 1)]);
        assert_eq!(yu, uy);
    }

    #[test]
    fn eq_rejects_subset_in_either_direction() {
        let small = Regressor::from(vec![Symbol::new("y", 1, 1)]);
        let big = Regressor::from(vec![Symbol::new("y", 1, 1), Symbol::new("u", 1, 1)]);
        assert_ne!(small, big);
        assert_ne!(big, small);
    }

    #[test]
    fn eq_distinguishes_different_powers() {
        let p2 = Regressor::from(vec![Symbol::new("y", 1, 2)]);
        let p3 = Regressor::from(vec![Symbol::new("y", 1, 3)]);
        assert_ne!(p2, p3);
    }

    #[test]
    fn build_regressors_222_has_fourteen_terms() {
        assert_eq!(
            build_regressors(HashMap::from([("y", 2), ("u", 2)]), 2).len(),
            14
        );
    }

    #[test]
    fn build_regressors_223_has_thirtyfour_terms() {
        assert_eq!(
            build_regressors(HashMap::from([("y", 2), ("u", 2)]), 3).len(),
            34
        );
    }

    #[test]
    fn build_regressors_each_term_has_total_power_at_most_n() {
        for r in build_regressors(HashMap::from([("y", 2), ("u", 2)]), 3) {
            let total: usize = r.terms().iter().map(|s| s.power()).sum();
            assert!(
                (1..=3).contains(&total),
                "total power {total} out of [1, 3]"
            );
        }
    }

    #[test]
    fn eval_returns_empty_for_regressor_with_no_terms() {
        let r = Regressor::from(vec![]);
        let samples: HashMap<&str, &[f32]> = HashMap::from([("y", [1.0, 2.0, 3.0].as_slice())]);
        assert_eq!(r.eval(&samples), Vec::<f32>::new());
    }

    #[test]
    fn eval_single_term_matches_symbol_eval() {
        // y(k-1) on [1,2,3,4]: defined at k=1,2,3 with values samples[0..3] = [1, 2, 3].
        let r = Regressor::new("y", 1);
        let samples: HashMap<&str, &[f32]> =
            HashMap::from([("y", [1.0, 2.0, 3.0, 4.0].as_slice())]);
        assert_eq!(r.eval(&samples), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn eval_single_term_with_power_squares_each_sample() {
        let r = Regressor::from(vec![Symbol::new("y", 0, 2)]);
        let samples: HashMap<&str, &[f32]> = HashMap::from([("y", [2.0, 3.0, 4.0].as_slice())]);
        assert_eq!(r.eval(&samples), vec![4.0, 9.0, 16.0]);
    }

    #[test]
    fn eval_with_equal_length_terms_multiplies_pointwise() {
        let r = Regressor::from(vec![Symbol::new("y", 0, 1), Symbol::new("u", 0, 1)]);
        let buf: &[f32] = &[2.0, 3.0, 4.0];
        let samples: HashMap<&str, &[f32]> = HashMap::from([("y", buf), ("u", buf)]);
        assert_eq!(r.eval(&samples), vec![4.0, 9.0, 16.0]);
    }

    #[test]
    fn eval_computes_lagged_product() {
        // y(k-1) * y(k-2) on [1, 2, 3, 4]:
        //   y(k-1).eval = [1, 2, 3];  y(k-2).eval = [1, 2]
        //   align (skip 1 from longer): [2, 3] * [1, 2] = [2, 6]
        // matches the NARMAX expectation: at k=2 → s1·s0 = 2; at k=3 → s2·s1 = 6.
        let r = Regressor::from(vec![Symbol::new("y", 1, 1), Symbol::new("y", 2, 1)]);
        let samples: HashMap<&str, &[f32]> =
            HashMap::from([("y", [1.0, 2.0, 3.0, 4.0].as_slice())]);
        assert_eq!(r.eval(&samples), vec![2.0, 6.0]);
    }

    #[test]
    fn eval_split_atoms_match_merged_power() {
        let split = Regressor::from(vec![Symbol::new("y", 0, 1), Symbol::new("y", 0, 1)]);
        let merged = Regressor::from(vec![Symbol::new("y", 0, 2)]);
        let samples: HashMap<&str, &[f32]> = HashMap::from([("y", [1.5, -2.0, 3.0].as_slice())]);
        assert_eq!(split.eval(&samples), merged.eval(&samples));
    }

    #[test]
    fn eval_returns_empty_when_any_term_index_exceeds_samples() {
        let r = Regressor::from(vec![Symbol::new("y", 0, 1), Symbol::new("y", 5, 1)]);
        let samples: HashMap<&str, &[f32]> = HashMap::from([("y", [1.0, 2.0].as_slice())]);
        assert_eq!(r.eval(&samples), Vec::<f32>::new());
    }

    #[test]
    fn eval_returns_empty_for_empty_samples() {
        let r = Regressor::new("y", 0);
        let samples: HashMap<&str, &[f32]> = HashMap::from([("y", [].as_slice())]);
        assert_eq!(r.eval(&samples), Vec::<f32>::new());
    }

    #[test]
    fn eval_three_terms_align_to_largest_index() {
        let r = Regressor::from(vec![
            Symbol::new("y", 0, 1),
            Symbol::new("y", 1, 1),
            Symbol::new("y", 2, 1),
        ]);
        // y(k) * y(k-1) * y(k-2) on [1,2,3,4,5]:
        //   y(k).eval   = [1,2,3,4,5]  (len 5)
        //   y(k-1).eval = [1,2,3,4]    (len 4)
        //   y(k-2).eval = [1,2,3]      (len 3)  — least_len = 3
        //   y(k)   skips 2 → [3,4,5]
        //   y(k-1) skips 1 → [2,3,4]
        //   y(k-2) skips 0 → [1,2,3]
        // product at k=2,3,4: [3·2·1, 4·3·2, 5·4·3] = [6, 24, 60]
        let samples: HashMap<&str, &[f32]> =
            HashMap::from([("y", [1.0, 2.0, 3.0, 4.0, 5.0].as_slice())]);
        assert_eq!(r.eval(&samples), vec![6.0, 24.0, 60.0]);
    }

    #[test]
    fn eval_with_power_zero_term_acts_as_identity() {
        let r = Regressor::from(vec![Symbol::new("y", 0, 0), Symbol::new("y", 0, 2)]);
        let samples: HashMap<&str, &[f32]> = HashMap::from([("y", [2.0, 3.0].as_slice())]);
        assert_eq!(r.eval(&samples), vec![4.0, 9.0]);
    }

    #[test]
    fn test_223() {
        let output = build_regressors(HashMap::from([("y", 2), ("u", 2)]), 3);
        let a = Regressor::new("y", 1);
        let b = Regressor::new("y", 2);
        let c = Regressor::new("u", 1);
        let d = Regressor::new("u", 2);

        let expected = vec![
            a.clone(),
            b.clone(),
            c.clone(),
            d.clone(),
            &a * &a,
            &a * &b,
            &a * &c,
            &a * &d,
            &b * &b,
            &b * &c,
            &b * &d,
            &c * &c,
            &c * &d,
            &d * &d,
            &a * &a * a.clone(),
            &a * &b * a.clone(),
            &a * &c * a.clone(),
            &a * &d * a.clone(),
            &b * &b * a.clone(),
            &b * &c * a.clone(),
            &b * &d * a.clone(),
            &c * &c * a.clone(),
            &c * &d * a.clone(),
            &d * &d * a.clone(),
            &b * &b * b.clone(),
            &b * &c * b.clone(),
            &b * &d * b.clone(),
            &c * &c * b.clone(),
            &c * &d * b.clone(),
            &d * &d * b.clone(),
            &c * &c * c.clone(),
            &c * &d * c.clone(),
            &d * &d * c.clone(),
            &d * &d * d.clone(),
        ];

        assert_same_regressor_set(&output, &expected);
    }
}
