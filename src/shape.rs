
// Type alias for typles of usize cuz redundant

/// More clean than (usize, usize)
pub type Shape2 = (usize, usize);
/// More clean than (usize, usize, usize, usize)
pub type Shape4 = (usize, usize, usize, usize);

/// More clean than (usize, usize)
pub type Coord2 = (usize, usize);

/// More clean than (usize, usize, usize, usize)
pub type Coord4 = (usize, usize, usize, usize);

#[derive(Copy, Clone)]
pub enum Shape {
    // Vector, axes: 0
    D1(usize), 
    // Matrix, axes: 0, 1
    D2(usize, usize), 
    // Cube, axes: 0, 1, 2
    D3(usize, usize, usize), 
    // Quaternion, axes: 0, 1, 2, 3
    D4(usize, usize, usize, usize),
    // D5, axes: 0, 1, 2, 3, 4 
    D5(usize, usize, usize, usize, usize), 
}

impl Shape {
    pub fn len (&self) -> usize {
        match self {
            Shape::D1(a) => *a,
            Shape::D2(a, b) => a * b,
            Shape::D3(a, b, c) => a * b * c,
            Shape::D4(a, b, c, d) => a * b * c * d,
            Shape::D5(a, b, c, d, e) => a * b * c * d * e,
        }
    }

    pub fn to_tuple (&self) -> (usize, usize, usize, usize, usize) {
        match self {
            Shape::D1(a) => (*a, 1, 1, 1, 1),
            Shape::D2(a, b) => (*a, *b, 1, 1, 1),
            Shape::D3(a, b, c) => (*a, *b, *c, 1, 1),
            Shape::D4(a, b, c, d) => (*a, *b, *c, *d, 1),
            Shape::D5(a, b, c, d, e) => (*a, *b, *c, *d, *d),
        }
    }
}

pub enum Axis {
    D1, D2, D3, D4, D5
}
