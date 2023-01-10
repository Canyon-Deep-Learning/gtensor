
use super::*;

use pretty_assertions::assert_eq;

#[test]
fn copy () {
    let mut a = Tensor::from_vec(2, 3, 1, 1, 
        vec![
            1.0, 4.0,
            2.0, 5.0,
            3.0, 6.0,
        ], 0
    );

    let mut b = Tensor::new(2, 3, 1, 1, 1);

    Tensor::copy(&mut a, &mut b);

    assert_eq!(
        a.data,
        b.data,
    );
}

#[test]
fn swap () {
    let mut a = Tensor::from_vec(2, 3, 1, 1, 
        vec![
            6.0, 3.0, 
            5.0, 2.0, 
            4.0, 1.0,
        ], 0
    );

    let mut b = Tensor::from_vec(2, 3, 1, 1, 
        vec![
            1.0, 6.0, 
            2.0, 5.0, 
            3.0, 4.0,
        ], 0
    );

    Tensor::swap(&mut a, &mut b);

    assert_eq!(
        a.data,
        vec![
            1.0, 6.0, 
            2.0, 5.0, 
            3.0, 4.0,
        ]
    );

    assert_eq!(
        b.data,
        vec![
            6.0, 3.0, 
            5.0, 2.0, 
            4.0, 1.0,
        ]
    );
}