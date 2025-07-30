use crate::{mat2::Mat2, vec::Vec2}; // Adjust this path
use approx::AbsDiffEq;
use std::f32::consts::FRAC_PI_2;

// Helper for comparing matrices with a small tolerance.
fn mat2_approx_eq(a: &Mat2, b: &Mat2) -> bool {
    a.col0.abs_diff_eq(&b.col0, 1e-5) && a.col1.abs_diff_eq(&b.col1, 1e-5)
}

#[test]
fn test_construction_and_constants() {
    let m = Mat2::new(Vec2::new(1., 2.), Vec2::new(3., 4.));
    assert_eq!(m.col0, Vec2::new(1., 2.));
    assert_eq!(m.col1, Vec2::new(3., 4.));
    assert_eq!(Mat2::default(), Mat2::IDENTITY);
    assert!(Mat2::NAN.is_nan());
}

#[test]
fn test_from_rows() {
    let m = Mat2::from_rows(Vec2::new(1., 2.), Vec2::new(3., 4.));
    assert_eq!(m.col0, Vec2::new(1., 3.));
    assert_eq!(m.col1, Vec2::new(2., 4.));
}

#[test]
fn test_from_angle_and_rotation() {
    let r90 = Mat2::from_angle(FRAC_PI_2);
    let v = Vec2::new(1., 0.);
    let rotated = r90 * v;
    assert!(rotated.abs_diff_eq(&Vec2::new(0., 1.), 1e-6));
    assert!((r90.extract_rotation() - FRAC_PI_2).abs() < 1e-6);
}

#[test]
fn test_from_scale_and_extraction() {
    let s = Mat2::from_scale(Vec2::new(2., -3.));
    let v = Vec2::new(5., 5.);
    let scaled = s * v;
    assert!(scaled.abs_diff_eq(&Vec2::new(10., -15.), 1e-6));
    assert!(s.extract_scale().abs_diff_eq(&Vec2::new(2., 3.), 1e-6)); // extract_scale is absolute
    assert_eq!(s.extract_scale_raw(), Vec2::new(2., -3.));
}

#[test]
fn test_determinant_inverse_and_invertibility() {
    let m = Mat2::new(Vec2::new(4., 2.), Vec2::new(7., 6.));
    assert_eq!(m.determinant(), 10.0);
    assert!(m.is_invertible());

    let inv = m.inverse().unwrap();
    let identity = m * inv;
    assert!(mat2_approx_eq(&identity, &Mat2::IDENTITY));

    let singular = Mat2::new(Vec2::new(1., 1.), Vec2::new(1., 1.));
    assert_eq!(singular.determinant(), 0.0);
    assert!(!singular.is_invertible());
    assert!(singular.inverse().is_none());
}

#[test]
fn test_transpose() {
    let m = Mat2::from_rows(Vec2::new(1., 2.), Vec2::new(3., 4.));
    let t = m.transpose();
    assert_eq!(t.col0, Vec2::new(1., 2.));
    assert_eq!(t.transpose(), m);
}

#[test]
fn test_row_and_col_access() {
    let m = Mat2::new(Vec2::new(1., 2.), Vec2::new(3., 4.));
    assert_eq!(m.col(0), Vec2::new(1., 2.));
    assert_eq!(m.col(1), Vec2::new(3., 4.));
    assert_eq!(m.row(0), Vec2::new(1., 3.));
    assert_eq!(m.row(1), Vec2::new(2., 4.));
}

#[test]
fn test_decomposition() {
    let r = Mat2::from_angle(0.5);
    let s = Mat2::from_scale(Vec2::new(2., 3.));
    let m = r * s;

    let (dec_s, dec_r) = m.decompose();
    assert!((dec_r - 0.5).abs() < 1e-6);
    assert!(dec_s.abs_diff_eq(&Vec2::new(2., 3.), 1e-6));
}

#[test]
fn test_orthonormalize_and_is_orthogonal() {
    let m = Mat2::from_angle(0.5) * Mat2::from_scale(Vec2::new(2., 3.));
    assert!(!m.is_orthogonal(1e-6));

    let ortho = m.orthonormalize();
    assert!(ortho.is_orthogonal(1e-6));
    assert!((ortho.determinant() - 1.0).abs() < 1e-6);
}

#[test]
fn test_operator_overloads() {
    let m1 = Mat2::new(Vec2::new(1., 2.), Vec2::new(3., 4.));
    let m2 = Mat2::new(Vec2::new(5., 6.), Vec2::new(7., 8.));

    // Add
    assert_eq!(m1 + m2, Mat2::new(Vec2::new(6., 8.), Vec2::new(10., 12.)));
    // Sub
    assert_eq!(m1 - m2, Mat2::new(Vec2::new(-4., -4.), Vec2::new(-4., -4.)));
    // Mul<f32>
    assert_eq!(m1 * 2.0, Mat2::new(Vec2::new(2., 4.), Vec2::new(6., 8.)));
    // Neg
    assert_eq!(-m1, Mat2::new(Vec2::new(-1., -2.), Vec2::new(-3., -4.)));
}

#[test]
fn test_assignment_operators() {
    let mut m1 = Mat2::new(Vec2::new(1., 2.), Vec2::new(3., 4.));
    let m2 = Mat2::new(Vec2::new(5., 6.), Vec2::new(7., 8.));

    m1 += m2;
    assert_eq!(m1, Mat2::new(Vec2::new(6., 8.), Vec2::new(10., 12.)));

    m1 -= m2;
    assert_eq!(m1, Mat2::new(Vec2::new(1., 2.), Vec2::new(3., 4.)));

    let mut m3 = Mat2::from_scale(Vec2::new(2., 2.));
    m3 *= m3;
    assert_eq!(m3, Mat2::from_scale(Vec2::new(4., 4.)));

    m3 /= 2.0;
    assert_eq!(m3, Mat2::from_scale(Vec2::new(2., 2.)));
}

#[test]
#[should_panic]
fn test_index_out_of_bounds() {
    let m = Mat2::IDENTITY;
    let _ = m[2]; // This should panic
}
