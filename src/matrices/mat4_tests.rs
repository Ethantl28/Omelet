use crate::{
    mat4::Mat4,
    quat::Quat,
    vec::{Vec3, Vec4},
};
use approx::AbsDiffEq;
use std::f32::consts::FRAC_PI_2;

// A helper function for comparing matrices with a small tolerance.
fn mat4_approx_eq(a: &Mat4, b: &Mat4) -> bool {
    a.col0.abs_diff_eq(&b.col0, 1e-5)
        && a.col1.abs_diff_eq(&b.col1, 1e-5)
        && a.col2.abs_diff_eq(&b.col2, 1e-5)
        && a.col3.abs_diff_eq(&b.col3, 1e-5)
}

#[test]
fn test_construction_and_constants() {
    let m = Mat4::new(
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    );
    assert_eq!(m.col0, Vec4::new(1.0, 0.0, 0.0, 0.0));
    assert_eq!(m, Mat4::IDENTITY);

    assert_eq!(Mat4::default(), Mat4::IDENTITY);
    assert_eq!(
        Mat4::ZERO,
        Mat4::new(Vec4::ZERO, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO)
    );
    assert_eq!(Mat4::IDENTITY.determinant(), 1.0);
    assert!(Mat4::NAN.is_nan());
}

#[test]
fn test_from_and_to_array_and_tuple() {
    let m = Mat4::from_rows(
        Vec4::new(1., 2., 3., 4.),
        Vec4::new(5., 6., 7., 8.),
        Vec4::new(9., 10., 11., 12.),
        Vec4::new(13., 14., 15., 16.),
    );

    // Column-major array
    let col_major_arr = m.to_array_col_major();
    assert_eq!(
        col_major_arr,
        [
            1., 5., 9., 13., 2., 6., 10., 14., 3., 7., 11., 15., 4., 8., 12., 16.
        ]
    );
    assert!(mat4_approx_eq(&m, &Mat4::from_array(col_major_arr)));

    // Row-major array
    let row_major_arr = m.to_array_row_major();
    assert_eq!(
        row_major_arr,
        [
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.
        ]
    );
    assert_eq!(m.transpose().to_array_col_major(), row_major_arr);
}

#[test]
fn test_transformation_constructors() {
    // Translation
    let t = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
    let p = t.transform_point3(Vec3::new(10.0, 10.0, 10.0));
    assert!(p.abs_diff_eq(&Vec3::new(11.0, 12.0, 13.0), 1e-6));

    // Scale
    let s = Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0));
    let v = s.transform_vector3(Vec3::new(1.0, 1.0, 1.0));
    assert!(v.abs_diff_eq(&Vec3::new(2.0, 3.0, 4.0), 1e-6));

    // Rotation from Quat
    let q = Quat::from_rotation_y(FRAC_PI_2);
    let r = Mat4::from_quat(q);
    let rotated_v = r.transform_vector3(Vec3::new(1.0, 0.0, 0.0));
    assert!(rotated_v.abs_diff_eq(&Vec3::new(0.0, 0.0, -1.0), 1e-6));
}

#[test]
fn test_from_trs() {
    let translation = Vec3::new(1.0, 2.0, 3.0);
    let rotation = Quat::from_rotation_y(FRAC_PI_2);
    let scale = Vec3::new(2.0, 3.0, 4.0);
    let m = Mat4::from_trs(translation, rotation, scale);

    let point = Vec3::new(1.0, 0.0, 0.0);
    let transformed = m.transform_point3(point);

    // Expected order: Scale, then Rotate, then Translate
    // Scale: (1,0,0) -> (2,0,0)
    // Rotate: (2,0,0) -> (0,0,-2)
    // Translate: (0,0,-2) -> (1,2,1)
    assert!(transformed.abs_diff_eq(&Vec3::new(1.0, 2.0, 1.0), 1e-6));
}

#[test]
fn test_look_at_and_look_to() {
    let eye = Vec3::new(0.0, 2.0, 5.0);
    let target = Vec3::new(0.0, 2.0, 0.0);
    let up = Vec3::new(0.0, 1.0, 0.0);
    let view_matrix = Mat4::look_at(eye, target, up);
    let view_matrix_to = Mat4::look_to(eye, target - eye, up);

    assert!(mat4_approx_eq(&view_matrix, &view_matrix_to));

    // A point at the camera's eye should be transformed to the origin.
    let transformed_eye = view_matrix.transform_point3(eye);
    assert!(transformed_eye.abs_diff_eq(&Vec3::ZERO, 1e-6));

    // The target point should be on the negative Z axis, at a distance of 5.
    let transformed_target = view_matrix.transform_point3(target);
    assert!(transformed_target.abs_diff_eq(&Vec3::new(0.0, 0.0, -5.0), 1e-6));
}

#[test]
fn test_determinant_and_inverse() {
    let m = Mat4::from_trs(
        Vec3::new(5.0, -2.0, 3.0),
        Quat::from_euler_angles(0.5, 1.2, -0.3),
        Vec3::new(1.0, 2.0, 0.5),
    );

    let det = m.determinant();
    assert!((det - (1.0 * 2.0 * 0.5)).abs() < 1e-5);

    let inv_m = m.inverse().unwrap();
    let identity = m * inv_m;

    assert!(mat4_approx_eq(&identity, &Mat4::IDENTITY));

    // Test non-invertible matrix
    let singular = Mat4::from_scale(Vec3::new(1.0, 0.0, 1.0));
    assert!(!singular.is_invertible());
    assert!(singular.inverse().is_none());
}

#[test]
fn test_perspective_projection() {
    let proj = Mat4::perspective(FRAC_PI_2, 16.0 / 9.0, 0.1, 100.0);

    // Point right in the center, inside the frustum
    let point_in_front = Vec3::new(0.0, 0.0, -10.0);
    let projected = proj.project_point3(point_in_front);
    assert!(projected.x.abs() < 1e-6);
    assert!(projected.y.abs() < 1e-6);
    assert!(projected.z > -1.0 && projected.z < 1.0);

    // Point on the near plane should map to z = -1 in OpenGL NDC
    let point_on_near = Vec3::new(1.0, 1.0, -0.1);
    let projected_near = proj.project_point3(point_on_near);
    assert!((projected_near.z - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_orthographic_projection() {
    let proj = Mat4::orthographic(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0);

    // Center point should map to origin
    let center = proj.project_point3(Vec3::new(0.0, 0.0, -50.0));
    assert!(center.x.abs() < 1e-6 && center.y.abs() < 1e-6);

    // Top-right corner should map to (1, 1)
    let top_right = proj.project_point3(Vec3::new(10.0, 10.0, -10.0));
    assert!((top_right.x - 1.0).abs() < 1e-6);
    assert!((top_right.y - 1.0).abs() < 1e-6);
}

#[test]
fn test_operator_overloads() {
    let m1 = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
    let m2 = Mat4::from_scale(Vec3::new(2.0, 2.0, 2.0));

    // Add
    let sum = m1 + m2;
    assert_eq!(sum.col0.x, 3.0);
    assert_eq!(sum.col3.w, 2.0);

    // Sub
    let diff = m1 - Mat4::IDENTITY;
    assert_eq!(diff.col0.x, 0.0);
    assert_eq!(diff.col3, Vec4::new(1.0, 2.0, 3.0, 0.0));

    // Mul<f32>
    let scaled = Mat4::IDENTITY * 5.0;
    assert_eq!(scaled.determinant(), 5.0_f32.powi(4));

    // Neg
    let neg = -Mat4::IDENTITY;
    assert_eq!(neg.col0, Vec4::new(-1.0, 0.0, 0.0, 0.0));

    // Mul<Vec4>
    let v = Vec4::new(1.0, 2.0, 3.0, 1.0);
    let transformed = m1 * v;
    assert_eq!(transformed, Vec4::new(2.0, 4.0, 6.0, 1.0));
}

#[test]
fn test_assignment_operators() {
    let mut m1 = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
    let m2 = Mat4::IDENTITY;

    m1 += m2;
    assert_eq!(m1.col0.x, 2.0);
    assert_eq!(m1.col3.w, 2.0);

    m1 -= m2;
    assert_eq!(m1.col0.x, 1.0);

    let mut m3 = Mat4::from_scale(Vec3::new(2.0, 2.0, 2.0));
    m3 *= m3;
    assert_eq!(m3.col0.x, 4.0);

    m3 /= 2.0;
    assert_eq!(m3.col0.x, 2.0);
}

#[test]
fn test_index_access() {
    let mut m = Mat4::from_rows(
        Vec4::new(1., 2., 3., 4.),
        Vec4::new(5., 6., 7., 8.),
        Vec4::new(9., 10., 11., 12.),
        Vec4::new(13., 14., 15., 16.),
    );
    assert_eq!(m[0], Vec4::new(1.0, 5.0, 9.0, 13.0));
    assert_eq!(m[1], Vec4::new(2.0, 6.0, 10.0, 14.0));

    m[2] = Vec4::ZERO;
    assert_eq!(m[2], Vec4::ZERO);
}

#[test]
#[should_panic]
fn test_index_out_of_bounds() {
    let m = Mat4::IDENTITY;
    let _ = m[4]; // This should panic
}
