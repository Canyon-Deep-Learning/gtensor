
use super::*;

//use pretty_assertions::assert_eq;

#[test]
fn convolve_1 () {
    
    let a = Tensor::new_fill(5, 7, 1, 1, 1.0, 0);
    let mut b = Tensor::new(5, 7, 1, 1, 1);

    let filter = Tensor::from_vec(3, 3, 1, 1, 
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0, 
            7.0, 8.0, 9.0,
        ]
    );

    Tensor::convolve(&a, &mut b, &filter, 0, 1, 0);

    assert_eq!(
        b.data,
        vec![
            28.0, 39.0, 39.0, 39.0, 24.0, 33.0, 45.0,
            45.0, 45.0, 27.0, 33.0, 45.0, 45.0, 45.0,
            27.0, 33.0, 45.0, 45.0, 45.0, 27.0, 33.0,
            45.0, 45.0, 45.0, 27.0, 33.0, 45.0, 45.0, 
            45.0, 27.0, 16.0, 21.0, 21.0, 21.0, 12.0,
        ]
    );
}