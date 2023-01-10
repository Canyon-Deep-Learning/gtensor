
use crate::tensor3::Tensor;
use crate::shape::Shape;

#[test]
pub fn test_im2row () { 

    let from = vec![
        1., 2., 3., 4.,
        5., 6., 7., 8., 
        9., 10., 11., 12., 
        13., 14., 15., 16.,

        17., 18., 19., 20., 
        21., 22., 23., 24., 
        25., 26., 27., 28., 
        29., 30., 31., 32., 

        33., 34., 35., 36., 
        37., 38., 39., 40., 
        41., 42., 43., 44., 
        45., 46., 47., 48.,
    ];

    let a = Tensor::from_vec(Shape::D3(4, 4, 3), from).unwrap();

    let kx = 2;
    let ky = 2;
    let stride = 1;
    let padx = 0;
    let pady = 0;

    let mut b = Tensor::new(Shape::D2(3 * kx * ky, 
        ((a.shape.0 + 2 * padx - kx) / stride + 1) * ((a.shape.1 + 2 * pady - ky) / stride + 1) * 1));

    Tensor::im2row(&a, &mut b, kx, ky, padx, pady, stride, stride);

    for row in 0..b.shape.0 {
        println!("  ");
        for col in 0..b.shape.1 {
            print!(" {} ", b[(row, col)])
        }
    }
}