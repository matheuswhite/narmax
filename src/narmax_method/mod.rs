pub mod common;
pub mod frols;
pub mod mgs;
pub mod semp;

use crate::{narmax_model::NarmaxModel, regressor::Regressor};

pub trait NarmaxMethod {
    fn identify(self, regressors: Vec<Regressor>, y: &[f32], u: &[f32]) -> NarmaxModel;
}
