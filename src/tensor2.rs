
use std::rc::Rc;
use std::cell::RefCell;

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

use rayon::prelude::*;

use anyhow::{Result, anyhow};

use crate::shape::Shape;

type TensorRef = Tensor;

pub struct Tensor {
    /// Raw Pointer array to an F32. 
    data: NonNull<f32>,

    /// The Shape of the Tensor. \
    /// Can be D1, D2, D3, D4, or D5. 
    shape: Shape,

    /// The number of elements in the Tensor.
    len: usize,
}

////////////////////////////////////////////////////////////////////////////////////
// Implementation Block for Tensor Initialization and Basic Augmentation
////////////////////////////////////////////////////////////////////////////////////

impl Tensor {
// ---------- // ------------------------------------------ // ---------- //
    /// Create a new Tensor using a Shape.
    /// Tensors use raw pointers to store data. 
    /// To pass this Tensor around, use Tensor.get_ref() to get an immutable reference.
    pub fn new (shape: Shape) -> Result<Self> {
        unsafe {
            let layout_data = Layout::array::<f32>(shape.len()).unwrap();
            let layout_dead = Layout::new::<f32>();
            if let Some(data) = NonNull::new(alloc(layout_data) as *mut f32) {
                let mut me = Self {
                    data: data,
                    shape,
                    len: shape.len()
                };

                for i in me.iter_mut() {
                    *i = 0.0;
                }

                Ok(me)
            } else {
                Err(anyhow!("Tensor Init Failure: Data is Null."))
            }
        }
    }
// ---------- // ------------------------------------------ // ---------- //
    /// Create a new Tensor using the data from an existing Vector.
    pub fn from_vec(shape: Shape, from: Vec<f32>) -> Result<Self> {
        if shape.len() != from.len() {
            return Err(anyhow!(
                "Tensor Init Failure: shape.len() != from.len(). Must be same length."
            ));
        }

        let mut me = Tensor::new(shape).unwrap();

        for (i, f) in me.iter_mut().zip(from.iter()) {
            *i = *f;
        }

        Ok(me)
    }
// ---------- // ------------------------------------------ // ---------- //
    /// Fill the Tensor with random values in a range.
    pub fn randomize (&mut self, max: f32, min: f32) {
        for i in self.iter_mut() {
            *i = max + (fastrand::f32() * (min - max) / f32::MAX); 
        }
    }
// ---------- // ------------------------------------------ // ---------- //
    /// Fill the Tensor with any scalar.
    pub fn fill (&mut self, scalar: f32) {
        for i in self.iter_mut() {
            *i = scalar;
        }
    }
// ---------- // ------------------------------------------ // ---------- //
}

////////////////////////////////////////////////////////////////////////////////////
// Implementation Block for Major Tensor Operations
////////////////////////////////////////////////////////////////////////////////////

impl Tensor {

}

////////////////////////////////////////////////////////////////////////////////////
// Implementation Block for Basic Tensor Operations
////////////////////////////////////////////////////////////////////////////////////

impl Tensor {

}

////////////////////////////////////////////////////////////////////////////////////
// Implementations of built-in Operations
////////////////////////////////////////////////////////////////////////////////////
use std::ops::{
    AddAssign, SubAssign, MulAssign, DivAssign, 
    Deref, DerefMut
};

// ---------- // ------------------------------------------ // ---------- //
impl AddAssign<f32> for Tensor {
    fn add_assign(&mut self, rhs: f32) {
        for i in self.iter_mut() {
            *i += rhs;
        }
    }
}
// ---------- // ------------------------------------------ // ---------- //
impl SubAssign<f32> for Tensor {
    fn sub_assign(&mut self, rhs: f32) {
        for i in self.iter_mut() {
            *i -= rhs;
        }
    }
}
// ---------- // ------------------------------------------ // ---------- //
impl MulAssign<f32> for Tensor {
    fn mul_assign(&mut self, rhs: f32) {
        for i in self.iter_mut() {
            *i *= rhs;
        }
    }
}
// ---------- // ------------------------------------------ // ---------- //
impl DivAssign<f32> for Tensor {
    fn div_assign(&mut self, rhs: f32) {
        for i in self.iter_mut() {
            *i /= rhs;
        }
    }
}
// ---------- // ------------------------------------------ // ---------- //
impl AddAssign<Tensor> for Tensor {
    fn add_assign(&mut self, rhs: Tensor) {
        for (l, r) in self.iter_mut().zip(rhs.iter()) {
            *l += *r;
        }
    }
}
// ---------- // ------------------------------------------ // ---------- //
impl SubAssign<Tensor> for Tensor {
    fn sub_assign(&mut self, rhs: Tensor) {
        for (l, r) in self.iter_mut().zip(rhs.iter()) {
            *l -= *r;
        }
    }
}
// ---------- // ------------------------------------------ // ---------- //
impl MulAssign<Tensor> for Tensor {
    fn mul_assign(&mut self, rhs: Tensor) {
        for (l, r) in self.iter_mut().zip(rhs.iter()) {
            *l *= *r;
        }
    }
}
// ---------- // ------------------------------------------ // ---------- //
impl DivAssign<Tensor> for Tensor {
    fn div_assign(&mut self, rhs: Tensor) {
        for (l, r) in self.iter_mut().zip(rhs.iter()) {
            *l /= *r;
        }
    }
}
// ---------- // ------------------------------------------ // ---------- //
impl Deref for Tensor {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        unsafe {
            std::slice::from_raw_parts(self.data.as_ptr(), self.shape.len())
        }
    }
}
// ---------- // ------------------------------------------ // ---------- //
impl DerefMut for Tensor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            std::slice::from_raw_parts_mut(self.data.as_ptr(), self.shape.len())
        }
    }
}
// ---------- // ------------------------------------------ // ---------- //
impl Into<Vec<f32>> for Tensor {
    fn into(self) -> Vec<f32> {
        todo!()
    }
}
// ---------- // ------------------------------------------ // ---------- //
impl ParallelIterator for Tensor {
    type Item = f32;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item> {
        todo!()
    }
}

