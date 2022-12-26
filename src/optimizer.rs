
use crate::tensor::Tensor;

pub enum Optimizer {
    Momentum {
        vdw: Tensor, beta: f32, lr: f32
    },

    RMSProp {
        vdw: Tensor, beta: f32, lr: f32
    },

    Adam {
        vdw: Tensor, sdw: Tensor, vsdw: Tensor,
        beta1: f32, beta2: f32, lr: f32, step: i32,
    },

    SGD {
        lr : f32
    },
}

impl Optimizer {
    pub fn optimize (&mut self, weight: &mut Tensor, delta: &Tensor) {

        match self {
        // ------- // ----------------------------------------------------- // ------- //
            Optimizer::Momentum { 
                vdw, beta, lr 
            } => {
                for (v, d) in vdw.iter_mut().zip(delta.iter()) {
                    *v = (*beta * *v) + (1.0 - *beta) * *d;
                }
        
                update(weight, &vdw, *lr);
            },
        // ------- // ----------------------------------------------------- // ------- //
            Optimizer::RMSProp { 
                vdw, beta, lr 
            } => {
                for (v, d) in vdw.iter_mut().zip(delta.iter()) {
                    // v = past / ((beta * past) + (1 - beta) * (del * del)) + small value to avoid dividing by zero
                    *v = *d / ((*beta * *v) + (1.0 - *beta) * (*d * *d)) + 0.000000000001;
                }
        
                update(weight, &vdw, *lr);
            },
        // ------- // ----------------------------------------------------- // ------- //
            Optimizer::Adam { 
                vdw, sdw, vsdw, 
                beta1, beta2, lr, step
            } => {
                for (v, (s, (vs, d))) in 
                vdw.iter_mut().zip(sdw.iter_mut().zip(vsdw.iter_mut().zip(delta.iter()))) {
        
                    // Momentum - Compute the Exp Weighted Avg of past grads
                    *v = (*beta1 * *v) + (1.0 - *beta1) * *d;
        
                    // RMSProp - Compute the Exp Weighted Avg of past squares of grads
                    *s = (*beta2 * *s) + (1.0 - *beta2) * (*d * *d);
        
                    // Combine Momentum and RMSProp
                    *vs = 
                        (*v / (1.0 - (f32::powi(*beta1, *step)))) / 
                        (f32::sqrt(*s / (1.0 - (f32::powi(*beta2, *step)))) 
                        + 0.00000000000001);
                }
        
                update(weight, &vsdw, *lr);
        
                *step += 1;
            },
        // ------- // ----------------------------------------------------- // ------- //
            Optimizer::SGD { 
                lr 
            } => {
                update(weight, delta, *lr);
            },
        // ------- // ----------------------------------------------------- // ------- //
        }

    }
}

/// Provides Optimizers with 
fn update (weight: &mut Tensor, delta: &Tensor, lr: f32) {
    // Normal Weight Tensor Update
    if weight.len() == delta.len() {
        for (w, d) in weight.iter_mut().zip(delta.iter()) {
            *w *= lr * *d
        }
    }
    // Bias Update for Convolution
    else if weight.cols != delta.cols {
        for dur in 0..delta.duration {
            for chan in 0..delta.channels {
                for row in 0..delta.rows {
                    for col in 0..delta.cols {
                        weight[chan] -= delta[(row, col, chan, dur)] * lr;
                    }
                }
            }
        }
    }
    // Normal Bias Update
    else {
        for row in 0..delta.rows {
            for col in 0..delta.cols {
                weight[col] -= delta[(row, col)] * lr; 
            }
        }
    }
}