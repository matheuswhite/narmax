use crate::regressor::Regressor;
use nalgebra::DMatrix;

#[derive(Debug)]
pub struct NarmaxModel {
    pub regressors: Vec<Regressor>,
    pub theta: Vec<f32>,
    pub phi: Option<DMatrix<f32>>,
    pub err: Option<Vec<f32>>,
    pub selected_indices: Option<Vec<usize>>,
}
