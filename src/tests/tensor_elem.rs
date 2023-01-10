
// # Testing Tensor-Tensor Element Wise Operations # //

use super::*;

use pretty_assertions::assert_eq;

#[test]
fn tensor_add_1 () {
    let mut a = Tensor::from_vec(3, 3, 1, 1, 
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ], 1
    );

    let b = Tensor::from_vec(3, 1, 1, 1, 
        vec![
            1.0, 2.0, 3.0,
        ], 0
    );

    Tensor::add(&mut a, &b);

    assert_eq!(
        a.data,
        vec![
            2.0, 4.0, 6.0,
            5.0, 7.0, 9.0,
            8.0, 10.0, 12.0,
        ]
    );
}

#[test]
fn tensor_add_2 () {
    let mut a = Tensor::from_vec(6, 3, 1, 1, 
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ], 1
    );

    let b = Tensor::from_vec(3, 1, 1, 1, 
        vec![
            1.0, 2.0, 3.0,
        ], 0
    );

    Tensor::add(&mut a, &b);

    assert_eq!(
        a.data,
        vec![
            2.0, 4.0, 6.0, 5.0, 7.0, 9.0,
            8.0, 10.0, 12.0, 11.0, 13.0, 15.0,
            14.0, 16.0, 18.0, 17.0, 19.0, 21.0, 
        ]
    );
}

#[test]
fn tensor_sub_1 () {
    let mut a = Tensor::from_vec(3, 3, 1, 1, 
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ], 1
    );

    let b = Tensor::from_vec(3, 1, 1, 1, 
        vec![
            1.0, 2.0, 3.0,
        ], 0
    );

    Tensor::sub(&mut a, &b);

    assert_eq!(
        a.data,
        vec![
            0.0, 0.0, 0.0,
            3.0, 3.0, 3.0,
            6.0, 6.0, 6.0,
        ]
    );
}

#[test]
fn tensor_sub_2 () {
    let mut a = Tensor::from_vec(6, 3, 1, 1, 
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ], 1
    );

    let b = Tensor::from_vec(3, 1, 1, 1, 
        vec![
            1.0, 2.0, 3.0,
        ], 0
    );

    Tensor::sub(&mut a, &b);

    assert_eq!(
        a.data,
        vec![
            0.0, 0.0, 0.0, 3.0, 3.0, 3.0,
            6.0, 6.0, 6.0, 9.0, 9.0, 9.0,
            12.0, 12.0, 12.0, 15.0, 15.0, 15.0,
        ]
    );
}

#[test]
fn tensor_mul_1 () {
    let mut a = Tensor::from_vec(3, 3, 1, 1, 
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ], 1
    );

    let b = Tensor::from_vec(3, 1, 1, 1, 
        vec![
            1.0, 2.0, 3.0,
        ], 0
    );

    Tensor::mul(&mut a, &b);

    assert_eq!(
        a.data,
        vec![
            1.0, 4.0, 9.0,
            4.0, 10.0, 18.0,
            7.0, 16.0, 27.0,
        ]
    );
}

#[test]
fn tensor_mul_2 () {
    let mut a = Tensor::from_vec(6, 3, 1, 1, 
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ], 1
    );

    let b = Tensor::from_vec(3, 1, 1, 1, 
        vec![
            1.0, 2.0, 3.0,
        ], 0
    );

    Tensor::mul(&mut a, &b);

    assert_eq!(
        a.data,
        vec![
            1.0, 4.0, 9.0, 4.0, 10.0, 18.0,
            7.0, 16.0, 27.0, 10.0, 22.0, 36.0,
            13.0, 28.0, 45.0, 16.0, 34.0, 54.0,
        ]
    );
}

#[test]
fn tensor_div_1 () {
    let mut a = Tensor::from_vec(3, 3, 1, 1, 
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ], 1
    );

    let b = Tensor::from_vec(3, 1, 1, 1, 
        vec![
            1.0, 2.0, 3.0,
        ], 0
    );

    Tensor::div(&mut a, &b);

    assert_eq!(
        a.data,
        vec![
            1.0, 1.0, 1.0,
            4.0, 2.5, 2.0,
            7.0, 4.0, 3.0,
        ]
    );
}

#[test]
fn tensor_div_2 () {
    let mut a = Tensor::from_vec(6, 3, 1, 1, 
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ], 1
    );

    let b = Tensor::from_vec(3, 1, 1, 1, 
        vec![
            1.0, 2.0, 3.0,
        ], 0
    );

    Tensor::div(&mut a, &b);

    assert_eq!(
        a.data,
        vec![
            1.0, 1.0, 1.0, 4.0, 2.5, 2.0,
            7.0, 4.0, 3.0, 10.0, 5.5, 4.0,
            13.0, 7.0, 5.0, 16.0, 8.5, 6.0,
        ]
    );
}