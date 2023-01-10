
use std::rc::Rc;
use std::cell::RefCell;

use crate::operators::*;
use crate::tensor::Tensor;
use crate::shape::*;
use crate::optimizer::*;
use crate::graph::Graph;

use anyhow::Result;

/// [Op] provides descriptions of Operators, which can be given to the [Graph] for initialization. \
/// Operators are what drives the math. Operators include Convolutions, Dense Layers, Activations, Pooling Layers, etc. 
/// These descriptors are also where the error checking happens, to make sure everything is orderly. 
#[derive(Clone)]
pub enum Op {
// ------- // --------------- // ------- //
    Dense(usize),
// ------- // --------------- // ------- //
    Conv { 
        filter_count: usize,
        kernel_rows: usize,
        kernel_cols: usize,
        padding: usize,
        stride: usize,
    },
// ------- // --------------- // ------- //
    MaxPool {
        kernel_rows: usize,
        kernel_cols: usize,
        stride: usize,
    },
// ------- // --------------- // ------- //
    AvgPool {
        kernel_rows: usize,
        kernel_cols: usize,
        stride: usize,
    },
// ------- // --------------- // ------- //
    Flatten,
// ------- // --------------- // ------- //
    ReLU,
// ------- // --------------- // ------- //
    Sigmoid,
// ------- // --------------- // ------- //
    Tanh,
// ------- // --------------- // ------- //
    Softmax,
// ------- // --------------- // ------- //
    OptOverride(Box<Op>, Opt),
// ------- // --------------- // ------- //
    Group(Vec<Op>),
// ------- // --------------- // ------- //
    Split(Vec<Op>),
// ------- // --------------- // ------- //
}

impl Op {
    pub(crate) fn create (
        &self, 
        graph: &mut Vec<Box<dyn Operator>>, 
        prev_size: &mut Shape4, 
        global_opt: Opt,
        tag: &mut OpTag,
    ) -> Result<()> {

        let mut prev_tag = tag.clone();

        match self {
        // -------- // ----------------------------------------------------- // ------- //
            Op::Dense(size) => {
                graph.push(Box::new(
                    Dense {
                        output: Rc::new(RefCell::new(Tensor::new2((prev_size.0, *size)))),
                        weight: Tensor::new2_random((prev_size.1, *size), 0.3, -0.3),
                        bias: Tensor::new2_random((1, *size), 0.3, -0.3),
                        input: Rc::new(RefCell::default()),
                        del_w: Tensor::new2((prev_size.1, *size)),
                        del_i: Rc::new(RefCell::new(Tensor::new2((prev_size.0, prev_size.1)))),
                        opt_w: global_opt.create((prev_size.1, *size, 1, 1)),
                        opt_b: global_opt.create((prev_size.0, prev_size.1, 1, 1)),
                    }
                ));

                *prev_size = (prev_size.0, *size, 1, 1);
                *tag = OpTag::D2;
            }
        // -------- // ----------------------------------------------------- // ------- //
            Op::Conv {
                filter_count, kernel_rows,
                kernel_cols, padding, stride,
            } => {
                graph.push(Box::new(
                    Convolution {
                        output: 
                        Rc::new(RefCell::new(Tensor::new4((
                            ((prev_size.0 - padding * 2) - *kernel_rows + 1) / *stride,
                            ((prev_size.1 - padding * 2) - *kernel_cols + 1) / *stride,
                            *filter_count, prev_size.3
                        )))),
                        input: Rc::new(RefCell::default()),
                        kernel: Tensor::new4_random((*kernel_rows, *kernel_cols, prev_size.2, *filter_count), 0.3, -0.3),
                        kernel_rows: *kernel_rows,
                        kernel_cols: *kernel_cols,
                        num_filters: *filter_count,
                        bias: Tensor::new4_random((1, 1, prev_size.2, 1), 0.3, -0.3),
                        del_w: Tensor::new4((*kernel_rows, *kernel_cols, prev_size.2, prev_size.3)),
                        del_i: Rc::new(RefCell::new(Tensor::new4((prev_size.0, prev_size.1, prev_size.2, prev_size.3)))),
                        stride: *stride,
                        padding: *padding,
                        opt_w: global_opt.create((*kernel_rows, *kernel_cols, prev_size.2, prev_size.3)),
                        opt_b: global_opt.create((1, 1, prev_size.2, 1)),
                    }
                ));

                *prev_size = (
                    ((prev_size.0 - padding * 2) - *kernel_rows + 1) / *stride,
                    ((prev_size.1 - padding * 2) - *kernel_cols + 1) / *stride,
                    *filter_count, prev_size.3
                );

                *tag = OpTag::D4;
            }
        // -------- // ----------------------------------------------------- // ------- //
            Op::MaxPool { 
                kernel_rows, kernel_cols, stride 
            } => {
                graph.push(Box::new(
                    MaxPool {
                        output:
                        Rc::new(RefCell::new(Tensor::new4((
                            (prev_size.0 - (kernel_rows - 1)) / stride,
                            (prev_size.1 - (kernel_cols - 1)) / stride,
                            prev_size.2, prev_size.3
                        )))),
                        delta: Rc::new(RefCell::new(Tensor::new4((prev_size.0, prev_size.1, prev_size.2, prev_size.3)))),
                        cached_deltas: Vec::with_capacity(prev_size.0 * prev_size.1 * prev_size.2 * prev_size.3),
                        input: Rc::new(RefCell::default()),
                        stride: *stride,
                        kernel_rows: *kernel_rows,
                        kernel_cols: *kernel_cols,
                    }
                ));

                *prev_size = (
                    (prev_size.0 - (kernel_rows - 1)) / stride,
                    (prev_size.1 - (kernel_cols - 1)) / stride,
                    prev_size.2, prev_size.3
                );

                *tag = OpTag::D4;
            }
        // -------- // ----------------------------------------------------- // ------- //
            Op::AvgPool {
                kernel_rows, kernel_cols, stride,
            } => {
                graph.push(Box::new(
                    AvgPool {
                        output:
                        Rc::new(RefCell::new(Tensor::new4((
                            (prev_size.0 - (kernel_rows - 1)) / stride,
                            (prev_size.1 - (kernel_cols - 1)) / stride,
                            prev_size.2, prev_size.3
                        )))),
                        delta: Rc::new(RefCell::new(Tensor::new4((prev_size.0, prev_size.1, prev_size.2, prev_size.3)))),
                        input: Rc::new(RefCell::default()),
                        stride: *stride,
                        kernel_rows: *kernel_rows,
                        kernel_cols: *kernel_cols,
                    }
                ));

                *prev_size = (
                    (prev_size.0 - (kernel_rows - 1)) / stride,
                    (prev_size.1 - (kernel_cols - 1)) / stride,
                    prev_size.2, prev_size.3
                );

                *tag = OpTag::D4;
            }
        // -------- // ----------------------------------------------------- // ------- //
            Op::Flatten => {
                graph.push(Box::new(
                    Flatten {
                        input: Rc::new(RefCell::default()),
                        delta: Rc::new(RefCell::default()),
                        shape_in: *prev_size,
                        shape_out: (prev_size.3, prev_size.0 * prev_size.1 * prev_size.2, 1, 1),
                    }
                ));

                *prev_size = (prev_size.3, prev_size.0 * prev_size.1 * prev_size.2, 1, 1);

                *tag = OpTag::Flatten;
            }
        // -------- // ----------------------------------------------------- // ------- //
            Op::ReLU => {
                graph.push(Box::new(
                    ReLU {
                        input: Rc::new(RefCell::default()),
                    }
                ));

                *tag = OpTag::Actv;
            }
        // -------- // ----------------------------------------------------- // ------- //
            Op::Sigmoid => {
                graph.push(Box::new(
                    Sigmoid {
                        input: Rc::new(RefCell::default()),
                    }
                ));

                *tag = OpTag::Actv;
            }
        // -------- // ----------------------------------------------------- // ------- //
            Op::Tanh => {
                graph.push(Box::new(
                    Tanh {
                        input: Rc::new(RefCell::default())      ,          
                    }
                ));

                *tag = OpTag::Actv;
            }
        // -------- // ----------------------------------------------------- // ------- //
            Op::Softmax => {
                graph.push(Box::new(
                    Softmax {
                        input: Rc::new(RefCell::default())
                    }
                ));

                OpTag::check_order_error(OpTag::D2, prev_tag)?;

                *tag = OpTag::Actv;
            }   
        // -------- // ----------------------------------------------------- // ------- //
            Op::OptOverride (op, opt) => {
                op.create(graph, prev_size, *opt, tag)?;
            }
        // -------- // ----------------------------------------------------- // ------- //
            Op::Group (group) => {
                // For each module in the group
                for op in group.iter() {
                    // Create the op, then change the tag to match
                    op.create(graph, prev_size, global_opt, tag)?;
                    // check that the new op matches the old op
                    OpTag::check_order_error(*tag, prev_tag)?;
                    // assign the previous tag to the old tag
                    prev_tag = tag.clone();
                }
            } 
        // -------- // ----------------------------------------------------- // ------- //
            Op::Split (paths) => {
                let mut graphs: Vec<Graph> = Vec::new();

                for path in paths.iter() {
                    graphs.push(Graph::new(
                        paths.clone(), prev_size.clone(), global_opt, tag.clone()
                    )?);
                }

                Split {
                    paths: graphs,
                    pool: 
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(paths.len()).build().unwrap(),
                };

                *tag = OpTag::Split;

                todo!()
            }
        // -------- // ----------------------------------------------------- // ------- //
        }

        OpTag::check_order_error(*tag, prev_tag)?;

        Ok(())
    }
}

use std::ops::Add;

impl Add<Op> for Op {
    type Output = Op;

    // Op + Op = Group(Vec<Op>)
    fn add(self, rhs: Op) -> Self::Output {
        let mut new_ops = Vec::new();
        match self {
            // if self is a group
            Op::Group(ops) => match rhs {
                // and rhs is a group
                Op::Group(other_ops) =>
                    new_ops = [&ops[..], &other_ops[..]].concat(),
                // and rhs is a single op
                _=> new_ops = [&ops[..], &vec![rhs]].concat()
            }
            // if self is a single op
            _ => match rhs {
                // and rhs is a group
                Op::Group(other_ops) => 
                    new_ops = [&vec![self], &other_ops[..]].concat(),
                // and rhs is a single op
                _=> new_ops = vec![self, rhs]
            }
        }
        Op::Group(new_ops)
    }
}

impl Add<Opt> for Op {
    type Output = Op;

    fn add(self, rhs: Opt) -> Self::Output {
        match self {
            Op::Group(ops) => {
                let mut new_ops: Vec<Op> = Vec::with_capacity(ops.len());
                for (old, new) in ops.iter().zip(new_ops.iter_mut()) {
                    *new = Op::OptOverride(Box::new(old.clone()), rhs);
                }
                Op::Group(new_ops)
            }
            _ => {
                Op::OptOverride(Box::new(self), rhs)
            }
        }
    }
}

/// Used for Error-Checking during the setup phase
/// of the Graph initialization.
#[derive(Clone, Copy)]
pub enum OpTag {
    Flatten,
    Split,
    Start, 
    Actv,
    End,
    D2,
    D4,
}

impl OpTag {
    /// Check if A (the current layer) is able to 
    /// precede B (the previous layer).
    pub fn check_order_error (a: OpTag, b: OpTag) -> Result<()> {
        match a {
            OpTag::Flatten => todo!(),
            OpTag::Split => todo!(),
            OpTag::Actv => todo!(),
            OpTag::D2 => todo!(),
            OpTag::D4 => todo!(),
            _ => { todo!() }
        }
    }
}


pub(crate) type LR = f32;
pub(crate) type Beta = f32;
pub(crate) type Beta1 = f32;
pub(crate) type Beta2 = f32;

/// Optimizer Descriptors
#[derive(Copy, Clone)]
pub enum Opt {
    Momentum(LR, Beta),
    RMSProp(LR, Beta),
    Adam(LR, Beta1, Beta2),
    SGD(LR),
}

impl Opt {
    pub fn create (&self, delta_size: Shape4) -> Optimizer {
        match self {
    // ------------- // ------------------------------ // ------------- //
            Opt::Momentum(lr, beta) => {
                Optimizer::Momentum {
                    vdw: Tensor::new4(delta_size),
                    beta: *beta, lr: *lr
                }
            },
    // ------------- // ------------------------------ // ------------- //
            Opt::RMSProp(lr, beta) => {
                Optimizer::RMSProp {
                    vdw: Tensor::new4(delta_size),
                    beta: *beta, lr: *lr
                }
            },
    // ------------- // ------------------------------ // ------------- //
            Opt::Adam(lr, beta1, beta2) => {
                Optimizer::Adam {
                    vdw: Tensor::new4(delta_size),
                    sdw: Tensor::new4(delta_size),
                    vsdw: Tensor::new4(delta_size),
                    beta1: *beta1, beta2: *beta2, lr: *lr, step: 0,
                }
            },
    // ------------- // ------------------------------ // ------------- //
            Opt::SGD(lr) => {
                Optimizer::SGD {
                    lr: *lr
                }
            },
    // ------------- // ------------------------------ // ------------- //
        }
    }
}