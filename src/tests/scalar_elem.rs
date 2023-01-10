
use super::*;

use pretty_assertions::assert_eq;

#[test]
fn add_scalar () {
    let mut a = Tensor::from_vec(3, 1, 1, 1, 
        vec![
            1.0, 2.0, 3.0,
        ], 0
    );

    a.add_scalar(5.0);

    assert_eq!(
        a.data,
        vec![6.0, 7.0, 8.0]
    );
}

#[test]
fn sub_scalar () {
    let mut a = Tensor::from_vec(3, 1, 1, 1, 
        vec![
            1.0, 2.0, 3.0,
        ], 0
    );

    a.sub_scalar(5.0);

    assert_eq!(
        a.data,
        vec![-4.0, -3.0, -2.0]
    );
}

#[test]
fn mul_scalar () {
    let mut a = Tensor::from_vec(3, 1, 1, 1, 
        vec![
            1.0, 2.0, 3.0,
        ], 0
    );

    a.mul_scalar(5.0);

    assert_eq!(
        a.data,
        vec![5.0, 10.0, 15.0]
    );
}

#[test]
fn div_scalar () {
    let mut a = Tensor::from_vec(3, 1, 1, 1, 
        vec![
            1.0, 2.0, 3.0,
        ], 0
    );

    a.div_scalar(5.0);

    assert_eq!(
        a.data,
        vec![0.2, 0.4, 0.6]
    );
}