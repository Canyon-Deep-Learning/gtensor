
use crate::*;
use crate::shape::{Shape2, Shape4};
use anyhow::Result;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub channels: usize,
    pub duration: usize,
}

impl Tensor {
    /// Create a new Tensor, filled with Zeros. 
    pub fn new4 (shape: Shape4) -> Self {
        Self {
            data: vec![0.0; shape.0 + shape.1 + shape.2 + shape.3],
            rows: shape.0, cols: shape.1, channels: shape.2, duration: shape.3,
        }
    }

    /// Create a new Tensor, filled with Zeros
    pub fn new2 (shape: Shape2) -> Self {
        Self {
            data: vec![0.0; shape.0 + shape.1],
            rows: shape.0, cols: shape.1, channels: 1, duration: 1,
        }
    }

    /// Create a new Tensor, filled with random values in a range. 
    pub fn new4_random (shape: Shape4, max: f32, min: f32) -> Self {
        let mut tensor = Self {
            data: vec![0.0; shape.0 + shape.1 + shape.2 + shape.3],
            rows: shape.0, cols: shape.1, channels: shape.2, duration: shape.3,
        };
        tensor.fill_random(max, min);
        tensor
    }

    /// Create a new Tensor, filled with random values in a range. 
    pub fn new2_random (shape: Shape2, max: f32, min: f32) -> Self {
        let mut tensor = Self {
            data: vec![0.0; shape.0 + shape.1],
            rows: shape.0, cols: shape.1, channels: 1, duration: 1,
        };
        tensor.fill_random(max, min);
        tensor
    }

    /// Create a new Tensor, filled with a scalar. 
    pub fn new4_fill (shape: Shape4, scalar: f32) -> Self {
        Self {
            data: vec![scalar; shape.0 + shape.1 + shape.2 + shape.3],
            rows: shape.0, cols: shape.1, channels: shape.2, duration: shape.3,
        }
    }

    /// Create a new Tensor, filled with a scalar. 
    pub fn new2_fill (shape: Shape2, scalar: f32) -> Self {
        Self {
            data: vec![scalar; shape.0 + shape.1],
            rows: shape.0, cols: shape.1, channels: 1, duration: 1,
        }
    }

    /// Create a new Tensor, moving all the values from an existing vector. 
    /// Size of vector must be the same as rows * cols. 
    pub fn from_vec (shape: Shape4, data: Vec<f32>) -> Self {
        // Check if size of the vector and the shape matches 
        if shape.0 + shape.1 + shape.2 + shape.3 != data.len() { 
            panic!("Invalid from_vec: Size does not match!"); 
        }

        Self {
            data: vec![0.0; shape.0 + shape.1 + shape.2 + shape.3],
            rows: shape.0, cols: shape.1, channels: shape.2, duration: shape.3,
        }
    }
}

impl Tensor {
    /// Get the shape of the Tensor
    pub fn get_shape (&self) -> Shape4 {
        (self.rows, self.cols, self.channels, self.duration)
    }

    /// The length of the Tensor, or the rows*cols*channels*duration
    pub fn len (&self) -> usize {
        self.rows * self.cols * self.channels * self.duration
    }

    /// Reshape a Tensor.  Note: new shape must be the same length as the old one. 
    pub fn reshape (&mut self, shape: Shape4) -> Result<()> {
        if (shape.0 * shape.1 * shape.2 * shape.3) == self.len() {
            return Err(anyhow!("Reshape Error: Cannot Reshape to a different length."));
        }

        self.rows = shape.0;
        self.cols = shape.1;
        self.channels = shape.2;
        self.duration = shape.3;

        Ok(())
    }

    /// Fill the tensor with some Scalar
    pub fn fill (&mut self, scalar: f32) {
        for i in self.iter_mut() {
            *i = scalar;
        }
    }

    /// Fill the tensor with random values, given a range. 
    pub fn fill_random (&mut self, max: f32, min: f32) {
        for i in self.iter_mut() {
            *i = max + (fastrand::f32() * (min - max) / f32::MAX);
        }
    }

    /// Iterate mutably over the data in the tensor. 
    pub fn iter_mut (&mut self) -> std::slice::IterMut<f32> {
        self.data.iter_mut()
    }

    /// Iterate immutably over the data in the tensor
    pub fn iter (&self) -> std::slice::Iter<f32> {
        self.data.iter()
    }

    /// Returns True if self and other are the same shape.
    pub fn same_shape (&self, other: &Tensor) -> bool {
        (self.rows == other.rows) &&
        (self.cols == other.cols) &&
        (self.channels == other.channels) &&
        (self.duration == other.duration)
    }

    /// Add a Scalar to every element of the Tensor
    pub fn add_scal (&mut self, scalar: f32) {
        for i in self.data.iter_mut() {
            *i += scalar;
        }
    }

    /// Sub a Scalar to every element of the Tensor
    pub fn sub_scal (&mut self, scalar: f32) {
        for i in self.data.iter_mut() {
            *i -= scalar;
        }
    }

    /// Mul a Scalar to every element of the Tensor
    pub fn mul_scal (&mut self, scalar: f32) {
        for i in self.data.iter_mut() {
            *i *= scalar;
        }
    }

    /// Div a Scalar to every element of the Tensor
    pub fn div_scal (&mut self, scalar: f32) {
        for i in self.data.iter_mut() {
            *i /= scalar;
        }
    }

    /// Apples the Square Root on every element of the Tensor
    pub fn sqrt (&mut self) {
        for i in self.data.iter_mut() {
            *i = f32::sqrt(*i);
        }
    }

    /// Computes the sum of all elements in the Tensor
    pub fn sum (&self) -> f32 {
        let mut sum = 0.0;
        for i in self.data.iter() {
            sum += i;
        }
        sum
    }

    /// Raises every element of the calling Tensor to the power of pow. 
    pub fn pow (&mut self, pow: i32) {
        for i in self.data.iter_mut() {
            *i = f32::powi(*i, pow);
        }
    }

    /// Performs e^z on every element z of the calling Tensor. 
    pub fn exp (&mut self) {
        for i in self.data.iter_mut() {
            *i = f32::exp(*i);
        }
    }

    /// Checks to see if a Tensor is 1 or 2-D
    pub fn is_2d (&self) -> bool {
        self.channels == 1 && self.duration != 1
    }

}

impl Tensor {
    /// General Matrix Multiplication
    pub fn matrixmultiply (transa: bool, A: &Tensor, transb: bool, B: &Tensor, C: &mut Tensor) -> Result<()> {
        let (m, k, rsa, csa) = 
            //             m       k      rsa   csa           m        k   rsa  csa
            if !transa {(A.rows, A.cols, A.cols, 1)} else {(A.cols, A.rows, 1, A.cols)};

        let (n, rsb, csb) =
            //             n       rsb  csb           n    rsb csb
            if !transb {(B.cols, B.cols, 1)} else {(B.rows, 1, k)};

        if !A.is_2d() || !B.is_2d() || !C.is_2d() {
            return Err(anyhow!("Matrix Multiply Error: Tensors must be 2-D! (channels and duration = 1)"))
        }

        unsafe {
            matrixmultiply::sgemm(
                m, k, n,
                1.0, A.data.as_ptr(), rsa as isize, csa as isize,
                B.data.as_ptr(), rsb as isize, csb as isize, 0.0,
                C.data.as_mut_ptr(), n as isize, 1
            );
        }

        Ok(())
    }

    /// A + B -> C.\
    /// A, B, and C have to be the same size. 
    pub fn add (A: &Tensor, B: &Tensor, C: &mut Tensor) -> Result<()> {
        if A.len() != B.len() || B.len() != C.len() {
            return Err(anyhow!("Element-Wise Addition Error: A, B, and C must be the same size."));
        }

        for (a, (b, c)) in A.iter().zip(B.iter().zip(C.iter_mut())) {
            *c = *a + *b;
        }

        Ok(())
    }

    /// A - B -> C.\
    /// A, B, and C have to be the same size. 
    pub fn sub (A: &Tensor, B: &Tensor, C: &mut Tensor) -> Result<()> {
        if A.len() != B.len() || B.len() != C.len() {
            return Err(anyhow!("Element-Wise Subtraction Error: A, B, and C must be the same size."));
        }

        for (a, (b, c)) in A.iter().zip(B.iter().zip(C.iter_mut())) {
            *c = *a - *b;
        }

        Ok(())
    }

    /// A * B -> C.\
    /// A, B, and C have to be the same size. 
    pub fn mul (A: &Tensor, B: &Tensor, C: &mut Tensor) -> Result<()> {
        if A.len() != B.len() || B.len() != C.len() {
            return Err(anyhow!("Element-Wise Multiplication Error: A, B, and C must be the same size."));
        }

        for (a, (b, c)) in A.iter().zip(B.iter().zip(C.iter_mut())) {
            *c = *a * *b;
        }

        Ok(())
    }

    /// A + B -> B.\
    /// Adds A to B into B.
    pub fn add_into (A: &Tensor, B: &mut Tensor) -> Result<()> {
        if A.len() != B.len() {
            return Err(anyhow!("Element-Wise Add_Into Error: A, and B must be the same size."));
        }

        for (a, b) in A.iter().zip(B.iter_mut()) {
            *b = *a - *b;
        }

        Ok(())
    }

    /// A + B -> B.\
    /// Adds A to B into B.
    pub fn sub_into (A: &Tensor, B: &mut Tensor) -> Result<()> {
        if A.len() != B.len() {
            return Err(anyhow!("Element-Wise Sub_Into Error: A, and B must be the same size."));
        }

        for (a, b) in A.iter().zip(B.iter_mut()) {
            *b = *a - *b;
        }

        Ok(())
    }

    /// A * B -> B.\
    /// Muls A to B into B.
    pub fn mul_into (A: &Tensor, B: &mut Tensor) -> Result<()> {
        if A.len() != B.len() {
            return Err(anyhow!("Element-Wise Mul_Into Error: A, and B must be the same size."));
        }

        for (a, b) in A.iter().zip(B.iter_mut()) {
            *b = *a * *b;
        }

        Ok(())
    }

    /// A / B -> B.\
    /// Divs A to B into B.
    pub fn div_into (A: &Tensor, B: &mut Tensor) -> Result<()> {
        if A.len() != B.len() {
            return Err(anyhow!("Element-Wise Div_Into Error: A, and B must be the same size."));
        }

        for (a, b) in A.iter().zip(B.iter_mut()) {
            *b = *a / *b;
        }

        Ok(())
    }

    /// Swaps the data values of two Tensors. \
    /// Only works if both A and B have the same length. \
    /// Returns an InvalidOp otherwise. 
    pub fn swap (a: &mut Tensor, b: &mut Tensor) {
        std::mem::swap(&mut a.data, &mut b.data);
    }

    /// Copies the values of A into B. \
    /// The Values of B are override by the values of A. A stays the same. 
    pub fn copy (A: &Tensor, B: &mut Tensor) -> Result<()> {
        if A.len() != B.len() {
            return Err(anyhow!("Reshape Error: Cannot Reshape to a different length."));
        }

        for (ai, bi) in A.iter().zip(B.iter_mut()) {
            *bi = *ai;
        }

        Ok(())
    }
}

// Convolution Operations
impl Tensor {
    /// Perform a Convolution over a Tensor using a Kernel.\
    /// A: The Input Tensor - Either an Image or a Feature Map.\
    /// B: The Output Tensor - must be of size ((0..a.rows + padding * 2) - kernel.rows + 1) / stride ( and same for cols).\
    /// Kernel: A tensor that represents the filters for this layer. Its channels is the depth, and duration is the number of filters.\
    /// Bias: The Tensor which contains the Biases. Should be the same length as the number of channels in B. \
    /// Stride: The amount of cells that are jumped for each step.\
    /// Padding: The amount of border zeros to add. \
    pub fn convolve (A: &Tensor, B: &mut Tensor, kernel: &Tensor, bias: &Tensor, stride: usize, padding: usize, dilation: usize) -> Result<()> {

        if stride == 0 {
            return Err(anyhow!("Convolution Error: Stride must be at least 1."))
        }

        if dilation == 0 {
            return Err(anyhow!("Convolution Error: Dilation must be at least 1."))
        }

        if bias.len() != A.channels || bias.len() != B.channels {
            return Err(anyhow!("Convolution Error: Bias Length must be the same as the channels in A and the channels in B."))
        }

        for dur in 0..A.duration {
            for filter in 0..kernel.duration {
                for chan in 0..A.channels {
                    for bx in 0..B.cols {
                        for by in 0..B.rows {
                            let mut sum = 0.0_f32;
                            // Take into account dilation
                            // ((kernel.rows - 1) * dilation) is the amount of empty space between indices due to dilation
                            for kx in (0..((kernel.cols) + (kernel.cols - 1) * dilation)).step_by(dilation + 1) {
                                for ky in (0..((kernel.rows) + (kernel.rows - 1) * dilation)).step_by(dilation + 1) {
                                    // Convert b + k into coordinates on a. 
                                    // Add to the sum the appropriate product
                                    sum += A[((((bx + kx) - padding) * (stride + 1)),(((by + ky) - padding) * (stride + 1)), chan, dur)] * kernel[(kx, ky, chan, filter)];
                                }
                            }
                            B[(bx, by, filter, dur)] = sum + bias[chan];
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Same as convolve, but rotates the kernel by 180 degrees. (top left is now bottom right)
    /// This operation is used for Backporogation with respect to the Input
    /// A: The Filter used in the forward convolution
    pub fn convolve_tr (A: &Tensor, B: &mut Tensor, kernel: &Tensor, stride: usize, padding: usize, dilation: usize) -> Result<()> {

        if stride == 0 {
            return Err(anyhow!("Convolution Error: Stride must be at least 1."))
        }

        if dilation == 0 {
            return Err(anyhow!("Convolution Error: Dilation must be at least 1."))
        }

        for dur in 0..A.duration {
            for filter in 0..kernel.duration {
                for chan in 0..A.channels {
                    for bx in 0..B.rows {
                        for by in 0..B.cols {
                            let mut sum = 0.0_f32;
                            // Take into account dilation
                            // ((kernel.rows - 1) * dilation) is the amount of empty space between indices due to dilation
                            for kx in (0..((kernel.rows) + (kernel.rows - 1) * dilation)).step_by(dilation + 1) {
                                for ky in (0..((kernel.cols) + (kernel.cols - 1) * dilation)).step_by(dilation + 1) {
                                    // Convert b + k into coordinates on a. 
                                    // Add to the sum the appropriate product.
                                    sum += A[((((bx + kx) - padding) * (stride + 1)),(((by + ky) - padding) * (stride + 1)), chan, dur)] * 
                                            kernel[(kernel.rows - kx - 1, kernel.cols - ky - 1, chan, filter)];
                                }
                            }
                            B[(bx, by, filter, dur)] = sum;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

use std::ops::{Index, IndexMut};

impl Index<usize> for Tensor {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

// For 2d - x + rows * y
impl Index<(usize, usize)> for Tensor {
    type Output = f32;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0 + self.rows * index.1]
    }
}

impl IndexMut<(usize, usize)> for Tensor {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0 + self.rows * index.1]
    }
}

// For 4d - index = x + y * D1 + z * D1 * D2 + t * D1 * D2 * D3;
impl Index<(usize, usize, usize, usize)> for Tensor {
    type Output = f32;
    fn index(&self, index: (usize, usize, usize, usize)) -> &Self::Output {
        &self.data[index.0 + index.1 * self.rows + index.2 * self.rows * self.cols + index.3 * self.rows * self.cols * self.channels]
    }
}

impl IndexMut<(usize, usize, usize, usize)> for Tensor {
    fn index_mut(&mut self, index: (usize, usize, usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0 + index.1 * self.rows + index.2 * self.rows * self.cols + index.3 * self.rows * self.cols * self.channels]
    }
}

impl Default for Tensor {
    fn default() -> Self {
        Self { 
            data: Default::default(), 
            rows: Default::default(), 
            cols: Default::default(), 
            channels: Default::default(), 
            duration: Default::default() 
        }
    }
}