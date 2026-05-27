#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol {
    name: String,
    index: usize,
    power: usize,
}

impl Symbol {
    pub fn new(name: impl AsRef<str>, index: usize, power: usize) -> Self {
        Self {
            name: name.as_ref().to_string(),
            index,
            power,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn power(&self) -> usize {
        self.power
    }

    pub fn set_power(&mut self, power: usize) {
        self.power = power;
    }

    pub fn eval(&self, samples: &[f32]) -> Vec<f32> {
        let samples_len = samples.len();
        samples
            .iter()
            .take(samples_len.saturating_sub(self.index()))
            .map(|s| s.powi(self.power as i32))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accepts_str_and_string() {
        let from_str = Symbol::new("y", 1, 1);
        let from_string = Symbol::new(String::from("y"), 1, 1);
        let from_ref_string = Symbol::new(&String::from("y"), 1, 1);
        assert_eq!(from_str, from_string);
        assert_eq!(from_str, from_ref_string);
    }

    #[test]
    fn getters_return_stored_values() {
        let s = Symbol::new("u", 3, 2);
        assert_eq!(s.name(), "u");
        assert_eq!(s.index(), 3);
        assert_eq!(s.power(), 2);
    }

    #[test]
    fn set_power_mutates_only_power() {
        let mut s = Symbol::new("y", 1, 1);
        s.set_power(5);
        assert_eq!(s.power(), 5);
        assert_eq!(s.name(), "y");
        assert_eq!(s.index(), 1);
    }

    #[test]
    fn eq_compares_all_three_fields() {
        let base = Symbol::new("y", 1, 2);
        assert_eq!(base, Symbol::new("y", 1, 2));
        assert_ne!(base, Symbol::new("u", 1, 2));
        assert_ne!(base, Symbol::new("y", 2, 2));
        assert_ne!(base, Symbol::new("y", 1, 3));
    }

    #[test]
    fn eval_index_zero_power_one_returns_samples_unchanged() {
        let s = Symbol::new("y", 0, 1);
        assert_eq!(s.eval(&[1.0, 2.0, 3.0]), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn eval_raises_each_sample_to_power() {
        let s = Symbol::new("y", 0, 2);
        assert_eq!(s.eval(&[1.0, 2.0, 3.0]), vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn eval_returns_prefix_of_length_len_minus_index() {
        // y(k-2) on a 4-sample buffer is defined at k=2,3 with values samples[0], samples[1].
        let s = Symbol::new("y", 2, 1);
        assert_eq!(s.eval(&[10.0, 20.0, 30.0, 40.0]), vec![10.0, 20.0]);
    }

    #[test]
    fn eval_combines_prefix_and_power() {
        // y(k-1)^3 on [10, 2, 3]: prefix is [10, 2], cubed → [1000, 8].
        let s = Symbol::new("y", 1, 3);
        assert_eq!(s.eval(&[10.0, 2.0, 3.0]), vec![1000.0, 8.0]);
    }

    #[test]
    fn eval_returns_empty_when_index_equals_len() {
        let s = Symbol::new("y", 3, 1);
        assert_eq!(s.eval(&[1.0, 2.0, 3.0]), Vec::<f32>::new());
    }

    #[test]
    fn eval_returns_empty_when_index_exceeds_len() {
        let s = Symbol::new("y", 10, 1);
        assert_eq!(s.eval(&[1.0, 2.0]), Vec::<f32>::new());
    }

    #[test]
    fn eval_returns_empty_for_empty_samples() {
        let s = Symbol::new("y", 0, 2);
        assert_eq!(s.eval(&[]), Vec::<f32>::new());
    }

    #[test]
    fn eval_with_power_zero_returns_ones() {
        let s = Symbol::new("y", 0, 0);
        assert_eq!(s.eval(&[2.0, -3.0, 0.5]), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn eval_handles_negative_samples() {
        let s = Symbol::new("y", 0, 3);
        assert_eq!(s.eval(&[-2.0, 2.0]), vec![-8.0, 8.0]);
    }

    #[test]
    fn eval_does_not_modify_name() {
        let s = Symbol::new("y", 1, 2);
        let _ = s.eval(&[1.0, 2.0, 3.0]);
        assert_eq!(s.name(), "y");
    }
}
