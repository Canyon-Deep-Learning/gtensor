
use std::cell::RefCell;
use std::rc::Rc;

use crate::tensor::Tensor;
use crate::operators::Operator;
use crate::descriptors::*;
use crate::shape::Shape4;

use anyhow::Result;

pub struct Graph {
    graph: Vec<Box<dyn Operator>>
}

impl Graph {
    pub fn new (descriptor: Vec<Op>, mut input_size: Shape4, global_opt: Opt, mut start_tag: OpTag) -> Result<Self> {

        let mut graph = Vec::new();

        for op in descriptor.iter() {
            op.create(&mut graph, &mut input_size, global_opt, &mut start_tag)?;
        }

        Ok(
            Self {
                graph,
            }
        )
    }

    pub fn forward (&mut self, input: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {

        let mut next = input;

        for (i, op) in self.graph.iter_mut().enumerate() {
            next = match op.forward(next) {
                Ok(v) => v,
                Err(e) => {
                    let name = op.get_name();
                    return Err(anyhow!(
                        "Graph Forward Step Error. Operator ID: '{i}'. Op name: '{name}'. Error Message: '{e}'"
                    ));
                }
            }
        }

        Ok(next)
    }

    pub fn backward (&mut self, delta: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {

        let mut next = delta;

        for (i, op) in self.graph.iter_mut().enumerate().rev() {
            next = match op.backward(next) {
                Ok(v) => v,
                Err(e) => {
                    let name = op.get_name();
                    return Err(anyhow!("Graph Backward Step Error. Operator ID: '{i}'. Op name: '{name}'. Error Message: '{e}'"));
                }
            }
        }

        Ok(next)
    }
}