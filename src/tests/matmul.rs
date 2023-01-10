
use super::*;

use pretty_assertions::assert_eq;


#[test]
fn matmul_1 () {

    let a = Tensor::from_vec(2, 3, 1, 1, 
        vec![
            1.0, 2.0, 3.0,
            1.0, 2.0, 3.0,
        ], 0
    );

    let mut b = Tensor::from_vec(3, 1, 1, 1, 
        vec![
            1.0,
            2.0,
            3.0,
        ], 1
    );

    let mut c = Tensor::from_vec(2, 1, 1, 1, 
        vec![
            1.0,
            2.0,
        ], 2
    );

    Tensor::multiply(false, &a, false, &mut b, &mut c);

    assert_eq!(
        c.data, 
        vec![
            14.0,
            14.0,
        ]
    );

}

#[test]
fn matmul_2 () {

    let a = Tensor::from_vec(2, 3, 1, 1, 
        vec![
            1.0, 2.0, 3.0,
            1.0, 2.0, 3.0,
        ], 0
    );

    let mut b = Tensor::from_vec(1, 3, 1, 1, 
        vec![
            1.0, 2.0, 3.0
        ], 1
    );

    let mut c = Tensor::from_vec(2, 1, 1, 1, 
        vec![
            1.0,
            2.0,
        ], 2
    );

    Tensor::multiply(false, &a, true, &mut b, &mut c);

    assert_eq!(
        c.data, 
        vec![
            14.0,
            14.0,
        ]
    );

}

#[test]
fn matmul_3 () {

    let a = Tensor::from_vec(3, 2, 1, 1, 
        vec![
            1.0, 1.0,
            2.0, 2.0,
            3.0, 3.0,
        ], 0
    );

    let mut b = Tensor::from_vec(3, 1, 1, 1, 
        vec![
            1.0,
            2.0,
            3.0,
        ], 1
    );

    let mut c = Tensor::from_vec(2, 1, 1, 1, 
        vec![
            1.0,
            2.0,
        ], 2
    );

    Tensor::multiply(true, &a, false, &mut b, &mut c);

    assert_eq!(
        c.data, 
        vec![
            14.0,
            14.0,
        ]
    );

}
