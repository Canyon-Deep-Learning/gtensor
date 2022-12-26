
#[macro_use]
extern crate anyhow;

extern crate serde;
use serde::{Serialize, Deserialize};

macro_rules! s {
    ($s:expr) => { $s.to_string() }
}

mod optimizer;
mod tensor;
mod graph;
mod operators;
mod shape;

/// Descriptors provide Enums which can be given to the [Graph] for graph initialization.
pub mod descriptors;

pub use graph::Graph;

pub use tensor::Tensor;

pub use shape::{Shape4, Shape2};