
use std::sync::{Arc, Mutex};
use std::rc::Rc;
use std::cell::RefCell;

use crate::tensor::Tensor;
use crate::optimizer::*;
use crate::shape::{Coord4, Shape4, Shape2};

use serde::{Serialize, Deserialize};
use anyhow::Result;

/// Defines Operator Behavior, the differentiable / Trainable Operations.
pub(crate) trait Operator {
    /// The Forward Pass of a Module
    /// Computes the output of the Module
    fn forward  (&mut self, input: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>>;

    /// The Backward Pass of a Module
    /// Computes the gradient of the Module
    /// If this layer has them, update the trainable params. 
    fn backward (&mut self, delta: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>>;

    /// Get the name of the Operator - mostly for Error Handling purposes.
    fn get_name (&self) -> &str;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct Dense {
    /// Holds the output of this layer in the Forward Step
    /// Found by the matrix multiplication of input x weight
    /// Batch Size x Layer Size.
    pub output: Rc<RefCell<Tensor>>,

    /// The Weights of the Connections to this layers' Neurons
    /// Prev Size x This Size
    pub weight: Tensor,

    /// The Bias of each Neuron
    /// 1 x This Size
    pub bias: Tensor,

    /// The Output of the previous layer
    /// Stored here as a reference because we use it in the forward
    /// pass and the backward pass for computing the gradient. 
    pub input: Rc<RefCell<Tensor>>,

    /// The Gradient with respect to the Weight
    /// Same size as Weight
    pub del_w: Tensor,

    /// The Gradient with respect to the Input
    /// Batch Size x Prev Size
    pub del_i: Rc<RefCell<Tensor>>,

    // The Optomizers for this Layer
    pub opt_w: Optimizer,
    pub opt_b: Optimizer,

    // initialization information
    pub size: usize,
}

impl Operator for Dense {
// ------- // ------------------------------------------------------------ // ------- //
    fn forward  (&mut self, input: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {
        // Cache the Input Reference here
        self.input = input;

        // MatMul the Input Matrix by the Weight Matrix to produce Output
        // (For each Neuron compute a line of n dimensions)
        Tensor::matrixmultiply(false, &self.input.borrow(), false, &self.weight, &mut self.output.borrow_mut())?;

        // Add the Bias to the Output Matrix
        // For each Neuron, add the base of the line computed.
        Tensor::add_into(&self.bias, &mut self.output.borrow_mut())?;

        // Send the Data forward to the next layer
        Ok(self.output.clone())
    }
// ------- // ------------------------------------------------------------ // ------- //
    fn backward (&mut self, delta: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {

        // Acquire Delta ref cell, since we'll use it more than once
        let del = delta.borrow();

        // Optimize the Bias 
        self.opt_b.optimize(&mut self.bias, &del);

        // Compute the Gradient wrt the Weights - resulting del_w is the same size as the weight tensor.
        Tensor::matrixmultiply(true, &self.input.borrow(), false, &del, &mut self.del_w)?;

        // Compute the Gradient wrt the Input - Resulting del_i is the same size as previous layer
        Tensor::matrixmultiply(false, &del, true, &mut self.weight, &mut self.del_i.borrow_mut())?;

        // Optomize the Weights
        self.opt_w.optimize(&mut self.weight, &self.del_w);

        Ok(self.del_i.clone())
    }
// ------- // ------------------------------------------------------------ // ------- //
    fn get_name (&self) -> &str {
        "Dense"
    }
// ------- // ------------------------------------------------------------ // ------- //
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct Convolution {
    /// The Output for this Layer.
    /// - rows: ((input.rows - padding * 2) - kernel.rows + 1) / stride
    /// - cols: ((input.cols - padding * 2) - kernel.cols + 1) / stride
    /// - channels: kernel.duration (number of filters)
    /// - duration: batch size
    pub output: Rc<RefCell<Tensor>>,

    /// The Output of the Previous Layer
    pub input: Rc<RefCell<Tensor>>,

    /// The Kernel for this Layer.
    /// Contains multiple Filters.
    /// - rows: custom 
    /// - cols: custom 
    /// - channels: a.channels
    /// - duration: custom (number of filters)
    pub kernel: Tensor,

    pub kernel_rows: usize,
    pub kernel_cols: usize,
    pub num_filters: usize,

    /// The Bias of each Channel.
    /// - rows: output.channels
    /// - cols: 1
    /// - channels: 1
    /// - duration: 1
    /// 
    /// Each element of bias corresponds to 1 channel of the output. 
    /// That element is added to that entire channel. 
    pub bias: Tensor,

    /// The Delta with respect to the Kernel.
    /// Same size as the Kernel. 
    pub del_w: Tensor,

    /// The Delta with respect to the input
    pub del_i: Rc<RefCell<Tensor>>,
    
    /// The Stride of the Kernel.
    /// How many cells it jumps for every step.
    /// Set to Zero for normal operation
    pub stride: usize,

    /// The Padding of the Kernel.
    /// The Amount of Zeroes placed around the border of the input Tensor.
    pub padding: usize,

    // The Optomizers for this Layer
    pub opt_w: Optimizer,
    pub opt_b: Optimizer,
}

impl Operator for Convolution {
// ------- // ------------------------------------------------------------ // ------- //
    fn forward  (&mut self, input: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {
        self.input = input;

        // Convolve across the Input with the Kernel into Output
        Tensor::convolve(
            &self.input.borrow(), &mut self.output.borrow_mut(), 
            &self.kernel, &self.bias, self.stride, self.padding, 1
        )?;

        Ok(self.output.clone())
    }
// ------- // ------------------------------------------------------------ // ------- //
    fn backward (&mut self, delta: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {

        // Acquire Output and Delta ref cells, since we'll use them more than once
        let out = self.output.borrow();
        let del = delta.borrow();

        // Optomize the bias with the delta wrt this layer
        self.opt_b.optimize(&mut self.bias, &del);

        // Compute the Gradient wrt the weights
        Tensor::convolve(
            &self.input.borrow(), &mut self.del_w, &delta.borrow(), 
            &self.bias, 1, self.padding, self.stride
        )?;

        // The Padding of the backward step wrt the inputs
        let back_pad = (((out.rows) + (out.rows - 1) * self.stride) - 1) + self.padding;

        // Compute the Gradient wrt the Input
        // Rotate the Kernel to be accurate with the _tr fn
        Tensor::convolve_tr(
            &self.kernel, &mut self.del_i.borrow_mut(), &del, 
            1, back_pad, self.stride
        )?;

        // Optomize the Weights
        // do it after the weights so we dont mess up the convolve_tr into input op
        self.opt_w.optimize(&mut self.kernel, &self.del_w);

        // Return del_i to be used by next layer
        Ok(self.del_i.clone())
    }
// ------- // ------------------------------------------------------------ // ------- //
    fn get_name (&self) -> &str {
        "Convolution"
    }
// ------- // ------------------------------------------------------------ // ------- //
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct AvgPool {
    /// The Output for this Layer
    pub output: Rc<RefCell<Tensor>>,

    /// The Delta wrt the Input
    pub delta: Rc<RefCell<Tensor>>,

    /// The Output of the Previous Layer
    pub input: Rc<RefCell<Tensor>>,

    /// The Stride for the Pooling
    /// 0 is normal operation
    pub stride: usize,

    // height / width of the filter
    pub kernel_rows: usize,
    pub kernel_cols: usize,
}

impl Operator for AvgPool {
// ------- // ------------------------------------------------------------ // ------- //
    fn forward  (&mut self, input: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {
        // Cache the Input for use in the backward step
        self.input = input;

        // Acquire the lock on Output mutably and input immutably
        let mut out = self.output.borrow_mut();
        let inp = self.input.borrow();
        
        // Max Pool Forward Operation
        for dur in 0..out.duration {
            for chan in 0..out.channels {
                for col in 0..out.cols {
                    for row in 0..out.rows {

                        let mut sum: f32 = 0.0;

                        for kx in 0..self.kernel_rows {
                            for ky in 0..self.kernel_cols {

                                let row = row * (self.stride + 1);
                                let col = col * (self.stride + 1);

                                sum += inp[(row + kx, col + ky, chan, dur)]
                            }
                        }
                        out[(row, col, chan, dur)] = sum / (self.kernel_rows * self.kernel_cols) as f32;
                    }
                }
            }
        }
        
        Ok(self.output.clone())
    }
// ------- // ------------------------------------------------------------ // ------- //
    fn backward (&mut self, delta: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {

        // Mutably Acquire self delta, since we are using multiple times. 
        let mut del_out = self.delta.borrow_mut();
        let del_in = delta.borrow();
        
        // Fill del self with zeros
        // Requires because the old values may not be overwritten
        del_out.fill(0.0);

        for dur in 0..del_in.duration {
            for chan in 0..del_in.channels {
                for col in 0..del_in.cols {
                    for row in 0..del_in.rows {
                        for kx in 0..self.kernel_rows {
                            for ky in 0..self.kernel_cols {
                                let row = row * (self.stride + 1);
                                let col = col * (self.stride + 1);

                                del_out[(row + kx, col + ky, chan, dur)] += 
                                    del_in[(row, col, chan, dur)] / (self.kernel_cols * self.kernel_rows) as f32;
                            }
                        }
                    }
                }
            }
        }

        Ok(self.delta.clone())
    }
// ------- // ------------------------------------------------------------ // ------- //
    fn get_name (&self) -> &str {
        "AvgPool"
    }
// ------- // ------------------------------------------------------------ // ------- //
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct MaxPool {
    /// The Output for this Layer
    pub output: Rc<RefCell<Tensor>>,

    /// The Delta wrt the Input
    pub delta: Rc<RefCell<Tensor>>,

    /// Cached max coordinates for efficiency (not benchmarked yet)
    pub cached_deltas: Vec<Shape4>,

    /// The Output of the Previous Layer
    pub input: Rc<RefCell<Tensor>>,

    /// The Stride for the Pooling
    /// 0 is normal operation
    pub stride: usize,

    // height / width of the filter
    pub kernel_rows: usize,
    pub kernel_cols: usize,
}

impl Operator for MaxPool {
// ------- // ------------------------------------------------------------ // ------- //
    fn forward  (&mut self, input: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {
        // Cache the Input for use in the backward step
        self.input = input;

        // Acquire the lock on Output mutably and input immutably
        let mut out = self.output.borrow_mut();
        let inp = self.input.borrow();
        
        // Max Pool Forward Operation
        for dur in 0..out.duration {
            for chan in 0..out.channels {
                for col in 0..out.cols {
                    for row in 0..out.rows {

                        let mut max: f32 = f32::MIN;
                        let mut coords: Coord4 = (0, 0, 0, 0);

                        for kx in 0..self.kernel_rows {
                            for ky in 0..self.kernel_cols {

                                let row = row * (self.stride + 1);
                                let col = col * (self.stride + 1);

                                if inp[(row + kx, col + ky, chan, dur)] > max {
                                    max = inp[(row + kx, col + ky, chan, dur)];
                                    coords = (row + kx, col + ky, chan, dur);
                                }
                            }
                        }
                        out[(row, col, chan, dur)] = max;
                        self.cached_deltas.push(coords);
                    }
                }
            }
        }
        
        Ok(self.output.clone())
    }
// ------- // ------------------------------------------------------------ // ------- //
    fn backward (&mut self, delta: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {

        // Mutably Acquire self delta, since we are using multiple times. 
        let mut del_self = delta.borrow_mut();

        // Set delta A to Zeros
        del_self.fill(0.0);

        // Perform the backward operation using the cached delta locations
        for (cache, in_del) in self.cached_deltas.iter().zip(delta.borrow().iter()) {
            // cached_deltas tells us where to add the delta values. in_del tells us what the delta values are. 
            del_self[*cache] += in_del; 
        } 

        // Reset cached_deltas (but not de-allocate)
        self.cached_deltas.clear();

        Ok(self.delta.clone())
    }
// ------- // ------------------------------------------------------------ // ------- //
    fn get_name (&self) -> &str {
        "MaxPool"
    }
// ------- // ------------------------------------------------------------ // ------- //
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

/// Flatten a 4-Dimensional Tensor into a 2-Dimensional Tensor. 
/// Used to convert from CNN data to Dense layer data.
pub struct Flatten {
    pub input: Rc<RefCell<Tensor>>,
    pub delta: Rc<RefCell<Tensor>>,

    // The shape of input when it enters "forward"
    pub shape_in: Shape4,
    // The shape of input when it exits "forward"
    pub shape_out: Shape4,
}

impl Operator for Flatten {
// ------- // ------------------------------------------------------------ // ------- //
    fn forward  (&mut self, input: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {
        
        self.input = input;

        // Reshape the Tensor to be flattened
        self.input.borrow_mut().reshape(self.shape_out)?;

        // Reflatten the Delta tensor
        self.delta.borrow_mut().reshape(self.shape_out)?;

        Ok(self.input.clone())
    }
// ------- // ------------------------------------------------------------ // ------- //
    fn backward (&mut self, delta: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {
        
        self.delta = delta;

        // Unflatten the input tensor
        self.input.borrow_mut().reshape(self.shape_in)?;

        // Unflatten the delta tensor
        self.delta.borrow_mut().reshape(self.shape_in)?;

        Ok(self.delta.clone())
    }
// ------- // ------------------------------------------------------------ // ------- //
    fn get_name (&self) -> &str {
        "Flatten"
    }
// ------- // ------------------------------------------------------------ // ------- //
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

use rayon::ThreadPool;

use crate::graph::Graph;

pub struct Split {
    pub paths: Vec<Graph>,
    pub pool: ThreadPool,
}

impl Operator for Split {
// ------- // ------------------------------------------------------------ // ------- //
    fn forward  (&mut self, input: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {
        for path in self.paths.iter_mut() {

        }

        todo!()
    }
// ------- // ------------------------------------------------------------ // ------- //
    fn backward (&mut self, delta: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {
        todo!()
    }
// ------- // ------------------------------------------------------------ // ------- //
    fn get_name (&self) -> &str {
        "Split"
    }
// ------- // ------------------------------------------------------------ // ------- //
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

/// A recent invention which stands for Rectified Linear Units. The formula is deceptively simple: max(0,z). 
/// Despite its name and appearance, it’s not linear and provides the same benefits as 
/// Sigmoid (i.e. the ability to learn nonlinear functions), but with better performance.
/// 
/// [More info on ReLU actv](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#relu)
pub struct ReLU {
    pub input: Rc<RefCell<Tensor>>,
}

impl Operator for ReLU {
// ------- // ------------------------------------------------------------ // ------- //
    fn forward  (&mut self, input: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {
        // Store the Input for later use
        self.input = input;

        // Perform the Relu Activation
        // Its ok to write into the input here. It wont be used until the next forward pass.
        for i in self.input.borrow_mut().iter_mut() {
            *i = if *i > 0.0 { *i } else { 0.0 }
        }

        Ok(self.input.clone())
    }
// ------- // ------------------------------------------------------------ // ------- //
    fn backward (&mut self, delta: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {
        // Multiply the deltas for this operation by the next layer deltas
        for (d, i) in delta.borrow_mut().iter_mut().zip(self.input.borrow().iter()) {
            if *i <= 0.0 { *d = 0.0 }
        } 

        Ok(delta)
    }
// ------- // ------------------------------------------------------------ // ------- //
    fn get_name (&self) -> &str {
        "ReLU"
    }
// ------- // ------------------------------------------------------------ // ------- //
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

/// Sigmoid takes a real value as input and outputs another value between 0 and 1. 
/// It’s easy to work with and has all the nice properties of activation functions: it’s non-linear, 
/// continuously differentiable, monotonic, and has a fixed output range.
/// 
/// [More info on Sigmoid actv](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#sigmoid)
pub struct Sigmoid {
    pub input: Rc<RefCell<Tensor>>,
}

impl Operator for Sigmoid {
// ------- // ------------------------------------------------------------ // ------- //
    fn forward  (&mut self, input: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {
        // Store the Input for later use
        self.input = input;

        // Perform the Sigmoid Activation
        // Its ok to write into the input here. It wont be used until the next forward pass.
        for i in self.input.borrow_mut().iter_mut() {
            *i = 1.0 / (1.0 + f32::exp(-*i));
        }

        Ok(self.input.clone())
    }
// ------- // ------------------------------------------------------------ // ------- //
    fn backward (&mut self, delta: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {
        // Multiply the deltas for this operation by the next layer deltas
        for (d, i) in delta.borrow_mut().iter_mut().zip(self.input.borrow().iter()) {
            *d = *i * (1.0 - *i);
        } 

        Ok(delta)
    }
// ------- // ------------------------------------------------------------ // ------- //
    fn get_name (&self) -> &str {
        "Sigmoid"
    }
// ------- // ------------------------------------------------------------ // ------- //
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

/// Softmax function calculates the probabilities distribution of the event over ‘n’ different events. 
/// In general way of saying, this function will calculate the probabilities of each target class 
/// over all possible target classes. Later the calculated probabilities will be helpful for 
/// determining the target class for the given inputs.
/// 
/// [more info on softmax actv](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#softmax) 
pub struct Softmax {
    pub input: Rc<RefCell<Tensor>>,
}

impl Operator for Softmax {
// ------- // ------------------------------------------------------------ // ------- //
    fn forward  (&mut self, input: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {
        // Store the Input for later use
        self.input = input;

        let mut inp = self.input.borrow_mut();
        
        for batch in 0..inp.rows {
            let mut sum: f32 = 0.0;
            for col in 0..inp.cols {
                inp[(batch, col)] = f32::exp(inp[(batch, col)]);
                sum += inp[(batch, col)];
            }

            for col in 0..inp.cols {
                inp[(batch, col)] = inp[(batch, col)] / sum;
            }
        }

        Ok(self.input.clone())
    }
// ------- // ------------------------------------------------------------ // ------- //
    fn backward (&mut self, delta: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {
        // Multiply the deltas for this operation by the next layer delta.

        let inp = self.input.borrow();
        let mut del = delta.borrow_mut();

        for batch in 0..inp.rows {
            for col in 0..inp.cols {
                del[(batch, col)] = inp[(batch, col)] * (1.0 - inp[(batch, col)]);
            }
        }

        Ok(delta.clone())
    }
// ------- // ------------------------------------------------------------ // ------- //
    fn get_name (&self) -> &str {
        "Softmax"
    }
// ------- // ------------------------------------------------------------ // ------- //
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

/// Tanh squashes a real-valued number to the range \[-1, 1\]. It’s non-linear. 
/// But unlike Sigmoid, its output is zero-centered. Therefore, in practice 
/// the tanh non-linearity is always preferred to the sigmoid nonlinearity.
/// 
/// [More info on Tanh Actv](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#tanh)
pub struct Tanh {
    pub input: Rc<RefCell<Tensor>>
}

impl Operator for Tanh {
    // ------- // ------------------------------------------------------------ // ------- //
        fn forward  (&mut self, input: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {
            // Store the Input for later use
            self.input = input;
    
            let mut inp = self.input.borrow_mut();

            // Perform the Tanh Activation
            // Its ok to write into the input here. It wont be used until the next forward pass.
            for i in inp.iter_mut() {
                *i = (f32::exp(*i) - f32::exp(-*i)) / (f32::exp(*i) + f32::exp(-*i));
            }
    
            Ok(self.input.clone())
        }
    // ------- // ------------------------------------------------------------ // ------- //
        fn backward (&mut self, delta: Rc<RefCell<Tensor>>) -> Result<Rc<RefCell<Tensor>>> {
            // Multiply the deltas for this operation by the next layer deltas

            let inp = self.input.borrow();
            let mut del = delta.borrow_mut();

            for (d, i) in del.iter_mut().zip(inp.iter()) {
                *d = 1.0 - f32::powi(*i, 2);
            } 
    
            Ok(delta.clone())
        }
// ------- // ------------------------------------------------------------ // ------- //
    fn get_name (&self) -> &str {
        "Tanh"
    }
// ------- // ------------------------------------------------------------ // ------- //
}

//////////////////////////////////////////////////////////////////////////////////////////////////////