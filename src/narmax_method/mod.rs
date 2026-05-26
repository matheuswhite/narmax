pub mod frols;

use crate::{narmax_model::NarmaxModel, regressor::Regressor};

pub trait NarmaxMethod {
    fn identify(self, regressors: Vec<Regressor>, y: &[f32], u: &[f32]) -> NarmaxModel;
}
