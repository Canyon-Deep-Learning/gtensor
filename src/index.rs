
// Local Includes
use super::Tensor;

// STL Includes
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

// Boolean Comparison

impl PartialEq for Tensor {
    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }

    /// Checks if the two tensors have the same shape
    fn eq(&self, other: &Self) -> bool {
        if self.rows == other.rows &&
           self.cols == other.cols &&
           self.channels == other.channels &&
           self.duration == other.duration 
        {
            true
        }
        else {
            false
        }
    }
}