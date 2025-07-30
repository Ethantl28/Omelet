use crate::{
    mat3::Mat3,
    vec::{Vec2, Vec3},
};
use approx::AbsDiffEq;
use std::f32::consts::FRAC_PI_2;

// Helper for comparing matrices with a small tolerance.
fn mat3_approx_eq(a: &Mat3, b: &Mat3) -> bool {
    a.col0.abs_diff_eq(&b.col0, 1e-5)
        && a.col1.abs_diff_eq(&b.col1, 1e-5)
        && a.col2.abs_diff_eq(&b.col2, 1e-5)
}

#[test]
fn test_construction() {
    let m = Mat3::new(
        Vec3::new(1., 2., 3.),
        Vec3::new(4., 5., 6.),
        Vec3::new(7., 8., 9.),
    );
    assert_eq!(m.col0, Vec3::new(1., 2., 3.));
    assert_eq!(m.col1, Vec3::new(4., 5., 6.));
    assert_eq!(m.col2, Vec3::new(7., 8., 9.));
    assert_eq!(Mat3::default(), Mat3::IDENTITY);
}

#[test]
fn test_from_rows() {
    let m = Mat3::from_rows(
        Vec3::new(1., 2., 3.),
        Vec3::new(4., 5., 6.),
        Vec3::new(7., 8., 9.),
    );
    assert_eq!(m.col0, Vec3::new(1., 4., 7.));
    assert_eq!(m.col1, Vec3::new(2., 5., 8.));
    assert_eq!(m.col2, Vec3::new(3., 6., 9.));
}

#[test]
fn test_transformation_constructors_2d() {
    // Translation
    let t = Mat3::from_translation(Vec2::new(10., 20.));
    let p = t.transform_point(Vec2::new(1., 2.));
    assert!(p.abs_diff_eq(&Vec2::new(11., 22.), 1e-6));

    // Scale
    let s = Mat3::from_scale(Vec2::new(2., 3.));
    let v = s.transform_vector(Vec2::new(5., 5.));
    assert!(v.abs_diff_eq(&Vec2::new(10., 15.), 1e-6));

    // Rotation
    let r = Mat3::from_angle_z(FRAC_PI_2);
    let rotated_v = r.transform_vector(Vec2::new(1., 0.));
    assert!(rotated_v.abs_diff_eq(&Vec2::new(0., 1.), 1e-6));

    // Shear
    let shear = Mat3::from_shear(Vec2::new(1., 0.));
    let sheared_p = shear.transform_point(Vec2::new(1., 1.));
    assert!(sheared_p.abs_diff_eq(&Vec2::new(2., 1.), 1e-6));
}

#[test]
fn test_from_trs() {
    let t = Vec2::new(10., 20.);
    let r = FRAC_PI_2;
    let s = Vec2::new(2., 2.);
    let p = Vec2::new(1., 1.); // Pivot

    let m = Mat3::from_trs(t, r, s, p);
    let point = Vec2::new(2., 1.); // Point relative to origin

    let transformed = m.transform_point(point);
    // Expected:
    // 1. Translate to pivot origin: (2,1) -> (1,0)
    // 2. Scale: (1,0) -> (2,0)
    // 3. Rotate: (2,0) -> (0,2)
    // 4. Translate back from pivot: (0,2) -> (1,3)
    // 5. Final translation: (1,3) -> (11, 23)
    assert!(transformed.abs_diff_eq(&Vec2::new(11., 23.), 1e-6));
}

#[test]
fn test_determinant_and_inverse() {
    let m = Mat3::from_scale(Vec2::new(2., 4.));
    assert_eq!(m.determinant(), 8.0);

    let inv = m.inverse().unwrap();
    let identity = m * inv;
    assert!(mat3_approx_eq(&identity, &Mat3::IDENTITY));

    let singular = Mat3::new(
        Vec3::new(1., 1., 0.),
        Vec3::new(1., 1., 0.),
        Vec3::new(0., 0., 1.),
    );
    assert!(!singular.is_invertible());
    assert!(singular.inverse().is_none());
}

#[test]
fn test_transpose() {
    let m = Mat3::from_rows(
        Vec3::new(1., 2., 3.),
        Vec3::new(4., 5., 6.),
        Vec3::new(7., 8., 9.),
    );
    let t = m.transpose();
    assert_eq!(t.col0, Vec3::new(1., 2., 3.));
    assert_eq!(t.transpose(), m);
}

#[test]
fn test_decomposition() {
    let t = Vec2::new(10., 20.);
    let r = 0.5;
    let s = Vec2::new(2., 3.);
    let m = Mat3::from_translation(t) * Mat3::from_angle_z(r) * Mat3::from_scale(s);

    let (dec_t, dec_r, dec_s) = m.decompose();
    assert!(dec_t.abs_diff_eq(&t, 1e-6));
    assert!((dec_r - r).abs() < 1e-6);
    assert!(dec_s.abs_diff_eq(&s, 1e-6));
}

#[test]
fn test_utility_checks() {
    let affine = Mat3::from_translation(Vec2::new(1., 2.));
    assert!(affine.is_affine());

    let not_affine = Mat3::new(
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1., 2., 3.),
    );
    assert!(!not_affine.is_affine());

    let mirroring = Mat3::from_scale(Vec2::new(-1., 1.));
    assert!(mirroring.has_mirroring());
    assert!(!affine.has_mirroring());
}

#[test]
fn test_operator_overloads() {
    let m1 = Mat3::from_translation(Vec2::new(1., 2.));
    let m2 = Mat3::from_scale(Vec2::new(2., 2.));

    // Add
    let sum = m1 + m2;
    assert_eq!(sum.col0.x, 3.0);
    assert_eq!(sum.col2.y, 2.0);

    // Sub
    let diff = m1 - Mat3::IDENTITY;
    assert_eq!(diff.col0.x, 0.0);
    assert_eq!(diff.col2, Vec3::new(1., 2., 0.));

    // Mul<f32>
    let scaled = Mat3::IDENTITY * 5.0;
    assert_eq!(scaled.determinant(), 125.0);

    // Neg
    let neg = -Mat3::IDENTITY;
    assert_eq!(neg.col0, Vec3::new(-1.0, 0.0, 0.0));

    // Mul<Vec3>
    let v = Vec3::new(1., 2., 1.);
    let transformed = m1 * v;
    assert_eq!(transformed, Vec3::new(2., 4., 1.));
}
