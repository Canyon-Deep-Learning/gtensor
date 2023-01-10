
use std::sync::Arc;

use rayon::prelude::*;

use anyhow::Result;

use crate::shape::{Axis, Shape};

pub struct Tensor {
    pub data: Vec<f32>,
    shape_type: Shape,
    pub shape: (usize, usize, usize, usize, usize)
}

////////////////////////////////////////////////////////////////////////////////////
// Implementation Block for Tensor Initialization and Basic Augmentation
////////////////////////////////////////////////////////////////////////////////////

impl Tensor {
// ---------- // ------------------------------------------ // ---------- //
    pub fn new (shape: Shape) -> Self {
        Self {
            data: vec![0.0; shape.len()],
            shape_type: shape, shape: shape.to_tuple()
        }
    }
// ---------- // ------------------------------------------ // ---------- //
    pub fn randomize (shape: Shape, max: f32, min: f32) -> Self {
        let mut data = Vec::with_capacity(shape.len());
        for i in data.iter_mut() {
            *i = max + (fastrand::f32() * (min - max) / f32::MAX); 
        }

        let (a, b, c, d, e) = shape.to_tuple();

        Self {
            data, shape_type: shape, shape: shape.to_tuple()
        }
    }
// ---------- // ------------------------------------------ // ---------- //
    /// Fill the Tensor with any scalar.
    pub fn new_fill (shape: Shape, scalar: f32) -> Self {
        let (a, b, c, d, e) = shape.to_tuple();

        Self {
            data: vec![scalar; shape.len()],
            shape_type: shape, shape: shape.to_tuple()
        }
    }
// ---------- // ------------------------------------------ // ---------- //
    /// Create a new Tensor using data from an existing Vector.
    pub fn from_vec (shape: Shape, from: Vec<f32>) -> Result<Self> {
        if shape.len() != from.len() {
            return Err(anyhow!("Tensor Init Error in from_vec: Shape.len() and From.len() must be the same."))
        }

        let (a, b, c, d, e) = shape.to_tuple();

        Ok(Self {
            data: from,
            shape_type: shape, shape: shape.to_tuple()
        })
    }
// ---------- // ------------------------------------------ // ---------- //
}

////////////////////////////////////////////////////////////////////////////////////
// Implementation Block for Major Tensor Operations
////////////////////////////////////////////////////////////////////////////////////

impl Tensor {
    /// gemm
    pub fn matrixmultiply (transa: bool, A: &Tensor, transb: bool, B: &Tensor, C: &mut Tensor) -> Result<()> {
        let (m, k, rsa, csa) = 
            //             m       k      rsa   csa           m        k   rsa  csa
            if !transa {(A.shape.0, A.shape.1, A.shape.1, 1)} else {(A.shape.1, A.shape.0, 1, A.shape.1)};

        let (n, rsb, csb) =
            //             n       rsb  csb           n    rsb csb
            if !transb {(B.shape.1, B.shape.1, 1)} else {(B.shape.0, 1, k)};

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

/*
float input[H][W][C];
float patches[H][W][K][K][C];
for ( h = 0; h < H; h++ )
    for ( w = 0; w < W; w++ )
        for ( kh = -K/2; kh < K/2; kh++ )
            for ( kw = -K/2; kw < K/2; kw++ )
                for ( c = 0; c < C; c++ )
                    patches[h][w][kh][kw][c] = input[h+kh][w+kw][c];

*/
    /// A: rows * cols * channels * duration
    /// B: [(channels * kx * ky), ((img rows + 2 * padding - kx) / stride + 1) * ((img cols + 2 * padding - ky) / stride + 1) * batch]
    /// K: kx * ky * channels * filter_count
    /// 
    /// B.rows = Number of kernel starting positions for every duration of A.
    /// B.cols = Each element is a patch of values 
    pub fn im2row (
        A: &Tensor, B: &mut Tensor, 
        kernelx: usize, kernely: usize, 
        padx: usize, pady: usize, 
        stridex: usize, stridey: usize, 
    ) {
        // number of starting positions on x-length and y-length
        let row: i32 = (((A.shape.0 + (2 * padx) - kernelx) / stridex) + 1) as i32;
        let col: i32 = (((A.shape.1 + (2 * pady) - kernely) / stridey) + 1) as i32;

        for (e, b) in B.data.iter_mut().enumerate() {
            // The Column of B we are in: aka the element of the kernel
            let by = (e % B.shape.1) as i32;
            // The Row of B we are in: aka the starting location of the kernel
            let bx = (e / B.shape.0) as i32;

            // The duration of A we are in / (number of starting positions)
            let ad = bx / (row * col);
            // The channel of A we are in 
            let ac = by / (kernelx * kernely) as i32;
            // The column of the Patch we are in
            let ay = by / kernely as i32;
            // The row of the patch we are in
            let ax = by % kernelx as i32;

            // the column of the starting position (top-left) of the patch, unadjusted
            let py = (bx / col) - (ad * col);
            // the row of the starting position (top-left) of the patch, unadjusted
            let px = (bx % row) - (ad * row);

            // The actual col position we are in
            let pya = ((py * stridey as i32) - pady as i32) + ay;
            // The actual row position we are in
            let pxa = ((px * stridex as i32) - padx as i32) + ax;

            // make sure we are not out of bounds
            if pya < 0 || pya >= A.shape.1 as i32 ||
               pxa < 0 || pxa >= A.shape.0 as i32 {
                *b = 0.0;
            } else {
                *b = A[(pxa as usize, pya as usize, ac as usize, ad as usize)];
            }
        }
    }

/* 
    /// A: rows * cols * channels
    /// B: rows * cols * kernel_rows * kernel_cols * channels
    /// K: kernel_rows * kernel_cols * channels
    /// xpad: Padding on the row side of A.
    /// ypad: Padding on the col side of A. 
    pub fn im2row (A: &Tensor, B: &mut Tensor, K: &Tensor, xpad: usize, ypad: usize, stride: usize) {
        for row in (-(xpad as i32)..(A.shape.0 + xpad) as i32).step_by(stride) {
            for col in (-(ypad as i32)..(A.shape.1 + ypad) as i32).step_by(stride) {
                for kr in -(K.shape.0 as i32 / 2)..(K.shape.0 as i32 / 2) {
                    for kc in -(K.shape.1 as i32 / 2)..(K.shape.1 as i32 / 2) {
                        for chan in 0..A.shape.2 {

                            if row > 0 && row < A.shape.0 as i32 &&
                               col > 0 && col < A.shape.1 as i32 &&
                               kr+K.shape.0 as i32/2 > 0 && kr+K.shape.0 as i32/2 < K.shape.0 as i32 &&
                               kc+K.shape.1 as i32/2 > 0 && kc+K.shape.1 as i32/2 < K.shape.1 as i32 &&
                               row+kr > 0 && row+kr < K.shape.0 as i32 &&
                               col+kc > 0 && col+kr < K.shape.1 as i32 &&
                               chan> 0 && chan< A.shape.2 {

                                B[(row as usize, col as usize, (kr + K.shape.0 as i32/2) as usize, (kc + K.shape.1 as i32/2) as usize, chan as usize)]
                                = A[((row + kr) as usize, (col + kc) as usize, chan as usize, 1)];
                            } else {
                                B[(row as usize, col as usize, kr as usize, kc as usize, chan as usize)] = 0.;
                            }
                        }
                    }
                }
            }
        }
    }
    */
}

////////////////////////////////////////////////////////////////////////////////////
// Implementation Block for Basic Tensor Operations
////////////////////////////////////////////////////////////////////////////////////

impl Tensor {
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

        B.data.par_iter_mut().zip(A.par_iter()).for_each(|(b, a)| *b = *a);

        Ok(())
    }

    /// Get the sum of every element in the Tensor
    pub fn sum (&self) -> f32 {
        self.data.par_iter().sum()
    }

    /// Sum over an axis in A into B. 
    /// - A: The Tensor being summed over. 
    /// - B: The target location for the sum. Must be a vector of the same size as the target axis. 
    /// - Axis: The axis to sum over. 
    /// 
    /// If you use Axis::Rows, each element of B (a vector) will contain the sum of the corresponding row in A. \
    /// If you use Axis::Cols, each element of B will contain the sum of the corresponding row in A. \
    /// 
    /// For Axis::Channels, each element of B will contain the sum of its corresponding channel in A. \
    /// For Axis::Duration, each element of B will contain the sum of its corresponding Duration is A. 
    /// 
    /// # Example
    /// ```
    /// // A is a 3x3 matrix (pseudocode)
    /// A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    /// // B is a 3x1 Tensor
    /// // B must be 3x1 or 1x3 since it must be the same length as the target axis on A. 
    /// B = Tensor::new2((3, 1));
    /// Tensor::sum_axis(&A, &mut B, Axis::D1);
    /// // B is now the sum of the rows
    /// B = [6, 15, 24] // B[0] = 1 + 2 + 3
    /// 
    /// Tensor::sum_axis(&A, &mut B, Axis::D2);
    /// // B is now the sum of the columns
    /// B = [12, 15, 18] // B[0] = 1 + 4 + 7
    pub fn sum_axis (A: &Tensor, B: &mut Tensor, axis: Axis) -> Result<()> {

        B.fill(0.0);

        match axis {
        // ----- // ------------------ // ----- //
            Axis::D1 => {
                if B.len() != A.shape.0 {
                    return Err(anyhow!("Sum Axis Failure: B must be the length of the axis summed!"));
                }

                for (i, a) in A.iter().enumerate() {
                    B[A.shape.0 % i] += a;
                }
            }
        // ----- // ------------------ // ----- //
            Axis::D2 => {
                if B.len() != A.shape.1 {
                    return Err(anyhow!("Sum Axis Failure: B must be the length of the axis summed!"));
                }

                for (i, a) in A.iter().enumerate() {
                    B[i % A.shape.1] += a;
                }
            },
        // ----- // ------------------ // ----- //
            Axis::D3 => {
                if B.len() != A.shape.2 {
                    return Err(anyhow!("Sum Axis Failure: B must be the length of the axis summed!"));
                }

                for (i, a) in A.iter().enumerate() {
                    B[(A.shape.0 * A.shape.1) % i] += a;
                }
            },
        // ----- // ------------------ // ----- //
            Axis::D4 => {
                if B.len() != A.shape.3 {
                    return Err(anyhow!("Sum Axis Failure: B must be the length of the axis summed!"));
                }

                for (i, a) in A.iter().enumerate() {
                    B[(A.shape.0 * A.shape.1 * A.shape.2) % i] += a;
                }
            },
        // ----- // ------------------ // ----- //
            Axis::D5 => {
                if B.len() != A.shape.4 {
                    return Err(anyhow!("Sum Axis Failure: B must be the length of the axis summed!"));
                }

                for (i, a) in A.iter().enumerate() {
                    B[(A.shape.0 * A.shape.1 * A.shape.2 * A.shape.3) % i] += a;
                }
            },
        // ----- // ------------------ // ----- //
        }

        Ok(())
    }

    /// A + B -> C.\
    /// A, B, and C have to be the same size. 
    pub fn add (A: &Tensor, B: &Tensor, C: &mut Tensor) -> Result<()> {
        if A.len() != B.len() || B.len() != C.len() {
            return Err(anyhow!("Element-Wise Addition Error: A, B, and C must be the same size."));
        }

        C.par_iter_mut().zip(B.par_iter().zip(A.par_iter())).for_each(
            |(c, (b, a))| { *c = *b + *a });

        Ok(())
    }

    /// A - B -> C.\
    /// A, B, and C have to be the same size. 
    pub fn sub (A: &Tensor, B: &Tensor, C: &mut Tensor) -> Result<()> {
        if A.len() != B.len() || B.len() != C.len() {
            return Err(anyhow!("Element-Wise Subtraction Error: A, B, and C must be the same size."));
        }

        C.par_iter_mut().zip(B.par_iter().zip(A.par_iter())).for_each(
            |(c, (b, a))| { *c = *b - *a });

        Ok(())
    }

    /// A * B -> C.\
    /// A, B, and C have to be the same size. 
    pub fn mul (A: &Tensor, B: &Tensor, C: &mut Tensor) -> Result<()> {
        if A.len() != B.len() || B.len() != C.len() {
            return Err(anyhow!("Element-Wise Multiplication Error: A, B, and C must be the same size."));
        }

        C.par_iter_mut().zip(B.par_iter().zip(A.par_iter())).for_each(
            |(c, (b, a))| { *c = *b * *a });

        Ok(())
    }

    /// A / B -> C.\
    /// A, B, and C have to be the same size. 
    pub fn div (A: &Tensor, B: &Tensor, C: &mut Tensor) -> Result<()> {
        if A.len() != B.len() || B.len() != C.len() {
            return Err(anyhow!("Element-Wise Division Error: A, B, and C must be the same size."));
        }

        C.par_iter_mut().zip(B.par_iter().zip(A.par_iter())).for_each(
            |(c, (b, a))| { *c = *b / *a });

        Ok(())
    }

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
        &self.data
    }
}
// ---------- // ------------------------------------------ // ---------- //
impl DerefMut for Tensor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}
// ---------- // ------------------------------------------ // ---------- //
impl Into<Vec<f32>> for Tensor {
    fn into(self) -> Vec<f32> {
        todo!()
    }
}
// ---------- // ------------------------------------------ // ---------- //
unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}
// ---------- // ------------------------------------------ // ---------- //
use std::ops::{Index, IndexMut};

// Index = xn ( D1 * ... * D{n-1} ) + x{n-1} ( D1 * ... * D{n-2} ) + ... + x2 * D1 + x1

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
// ---------- // ------------------------------------------ // ---------- //
// For 2d - x + rows * y
impl Index<(usize, usize)> for Tensor {
    type Output = f32;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0 + self.shape.0* index.1]
    }
}
impl IndexMut<(usize, usize)> for Tensor {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0 + self.shape.0 * index.1]
    }
}
// ---------- // ------------------------------------------ // ---------- //
impl Index<(usize, usize, usize)> for Tensor {
    type Output = f32;

    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        &self.data[index.2 * self.shape.0 * self.shape.1 + index.1 * self.shape.0 + index.0]
    }
}
impl IndexMut<(usize, usize, usize)> for Tensor {
    fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
        &mut self.data[index.2 * self.shape.0 * self.shape.1 + index.1 * self.shape.0 + index.0]
    }
}
// ---------- // ------------------------------------------ // ---------- //
// For 4d - index = x + y * D1 + z * D1 * D2 + t * D1 * D2 * D3;
impl Index<(usize, usize, usize, usize)> for Tensor {
    type Output = f32;
    fn index(&self, index: (usize, usize, usize, usize)) -> &Self::Output {
        &self.data[index.0 + index.1 * self.shape.0 + index.2 * self.shape.0 * self.shape.1 + index.3 * self.shape.0 * self.shape.1 * self.shape.2]
    }
}
impl IndexMut<(usize, usize, usize, usize)> for Tensor {
    fn index_mut(&mut self, index: (usize, usize, usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0 + index.1 * self.shape.0 + index.2 * self.shape.0 * self.shape.1 + index.3 * self.shape.0 * self.shape.1 * self.shape.2]
    }
}
// ---------- // ------------------------------------------ // ---------- //
// Index = xn ( D1 * ... * D{n-1} ) + x{n-1} ( D1 * ... * D{n-2} ) + ... + x2 * D1 + x1
// ---------- // ------------------------------------------ // ---------- //
impl Index<(usize, usize, usize, usize, usize)> for Tensor {
    type Output = f32;

    fn index(&self, index: (usize, usize, usize, usize, usize)) -> &Self::Output {
        &self.data[index.4 * self.shape.0 * self.shape.1 * self.shape.2 * self.shape.3 + index.3 * self.shape.0 * 
        self.shape.1 * self.shape.2 + index.2 * self.shape.0 * self.shape.1 + index.1 * self.shape.0 + index.0]
    }
}
impl IndexMut<(usize, usize, usize, usize, usize)> for Tensor {
    fn index_mut(&mut self, index: (usize, usize, usize, usize, usize)) -> &mut Self::Output {
        &mut self.data[index.4 * self.shape.0 * self.shape.1 * self.shape.2 * self.shape.3 + index.3 * self.shape.0 * 
        self.shape.1 * self.shape.2 + index.2 * self.shape.0 * self.shape.1 + index.1 * self.shape.0 + index.0]
    }
}
// ---------- // ------------------------------------------ // ---------- //
impl Default for Tensor {
    fn default() -> Self {
        Self { 
            data: Default::default(),
            shape_type: Shape::D1(1),
            shape: (1, 1, 1, 1, 1), 
        }
    }
}
// ---------- // ------------------------------------------ // ---------- //