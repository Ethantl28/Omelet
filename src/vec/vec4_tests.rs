
use approx::AbsDiffEq;
use std::f32::consts::FRAC_PI_2;
use crate::vec::{Vec3, Vec4}; // Adjust this path

#[test]
fn test_construction_and_constants() {
    let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(v.x, 1.0);
    assert_eq!(v.y, 2.0);
    assert_eq!(v.z, 3.0);
    assert_eq!(v.w, 4.0);

    assert_eq!(Vec4::ZERO, Vec4::new(0.0, 0.0, 0.0, 0.0));
    assert_eq!(Vec4::X, Vec4::new(1.0, 0.0, 0.0, 0.0));
    assert_eq!(Vec4::Y, Vec4::new(0.0, 1.0, 0.0, 0.0));
    assert_eq!(Vec4::Z, Vec4::new(0.0, 0.0, 1.0, 0.0));
    assert_eq!(Vec4::W, Vec4::new(0.0, 0.0, 0.0, 1.0));
    assert!(Vec4::NAN.is_nan());
}

#[test]
fn test_conversion() {
    let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(v.to_array(), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(Vec4::from_array([1.0, 2.0, 3.0, 4.0]), v);
    assert_eq!(v.to_tuple(), (1.0, 2.0, 3.0, 4.0));
    assert_eq!(Vec4::from_tuple((1.0, 2.0, 3.0, 4.0)), v);
    assert_eq!(Vec4::from_vec3_w(Vec3::new(1.0, 2.0, 3.0), 4.0), v);
    assert_eq!(v.xyz(), Vec3::new(1.0, 2.0, 3.0));
}

#[test]
fn test_math_utilities() {
    let v = Vec4::new(-1.0, 2.0, -3.0, 0.0);
    assert_eq!(v.abs(), Vec4::new(1.0, 2.0, 3.0, 0.0));
    assert_eq!(v.signum(), Vec4::new(-1.0, 1.0, -1.0, 0.0));
    assert_eq!(v.clamp(-0.5, 1.5), Vec4::new(-0.5, 1.5, -0.5, 0.0));
    assert_eq!(
        v.min(Vec4::new(-2.0, 3.0, -2.0, 1.0)),
        Vec4::new(-2.0, 2.0, -3.0, 0.0)
    );
    // Corrected the expected result for the max function's w component
    assert_eq!(
        v.max(Vec4::new(-2.0, 1.0, -4.0, -1.0)),
        Vec4::new(-1.0, 2.0, -3.0, 0.0)
    );
}

#[test]
fn test_magnitude_and_normalization() {
    let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    assert!((v.length() - (30.0_f32).sqrt()).abs() < 1e-6);
    assert_eq!(v.squared_length(), 30.0);

    let norm = v.normalize();
    assert!(norm.is_normalized());

    assert!(v.try_normalize().is_some());
    assert!(Vec4::ZERO.try_normalize().is_none());
    assert_eq!(Vec4::ZERO.normalize_or_zero(), Vec4::ZERO);
}

#[test]
fn test_dot_cross_and_angle() {
    let v1 = Vec4::X;
    let v2 = Vec4::Y;
    assert_eq!(v1.dot(v2), 0.0);
    assert_eq!(v1.cross_xyz(v2), Vec4::Z);
    assert!((Vec4::angle_between_radians(v1, v2) - FRAC_PI_2).abs() < 1e-6);
}

#[test]
fn test_interpolation() {
    let v1 = Vec4::new(0.0, 0.0, 0.0, 0.0);
    let v2 = Vec4::new(10.0, -10.0, 20.0, -20.0);
    assert!(
        v1.lerp(v2, 0.5)
            .abs_diff_eq(&Vec4::new(5.0, -5.0, 10.0, -10.0), 1e-6)
    );
}

#[test]
fn test_projection_and_reflection() {
    let v = Vec4::new(3.0, 4.0, 0.0, 0.0);
    let onto = Vec4::X;
    assert!(
        v.project(onto)
            .abs_diff_eq(&Vec4::new(3.0, 0.0, 0.0, 0.0), 1e-6)
    );
    assert!(
        v.reject(onto)
            .abs_diff_eq(&Vec4::new(0.0, 4.0, 0.0, 0.0), 1e-6)
    );

    let incident = Vec4::new(1.0, -1.0, 0.0, 0.0);
    let normal = Vec4::Y;
    assert!(
        incident
            .reflect(normal)
            .abs_diff_eq(&Vec4::new(1.0, 1.0, 0.0, 0.0), 1e-6)
    );
}

#[test]
fn test_distance_and_movement() {
    let v1 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let v2 = Vec4::new(4.0, 6.0, 3.0, 4.0); // dist = 5
    assert!((v1.distance(v2) - 5.0).abs() < 1e-6);
    assert_eq!(v1.squared_distance(v2), 25.0);

    let moved = Vec4::move_towards(v1, v2, 2.0);
    assert!(moved.abs_diff_eq(&Vec4::new(2.2, 3.6, 3.0, 4.0), 1e-6));

    let moved_fully = Vec4::move_towards(v1, v2, 10.0);
    assert_eq!(moved_fully, v2);
}

#[test]
fn test_operator_overloads() {
    let v1 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let v2 = Vec4::new(5.0, 6.0, 7.0, 8.0);

    assert_eq!(v1 + v2, Vec4::new(6.0, 8.0, 10.0, 12.0));
    assert_eq!(v1 - v2, Vec4::new(-4.0, -4.0, -4.0, -4.0));
    assert_eq!(v1 * v2, Vec4::new(5.0, 12.0, 21.0, 32.0));
    assert_eq!(v1 * 2.0, Vec4::new(2.0, 4.0, 6.0, 8.0));
    assert_eq!(v1 / 2.0, Vec4::new(0.5, 1.0, 1.5, 2.0));
    assert_eq!(-v1, Vec4::new(-1.0, -2.0, -3.0, -4.0));
}

#[test]
fn test_assignment_operators() {
    let mut v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    v += Vec4::new(5.0, 6.0, 7.0, 8.0);
    assert_eq!(v, Vec4::new(6.0, 8.0, 10.0, 12.0));
    v -= Vec4::new(1.0, 1.0, 1.0, 1.0);
    assert_eq!(v, Vec4::new(5.0, 7.0, 9.0, 11.0));
    v *= 2.0;
    assert_eq!(v, Vec4::new(10.0, 14.0, 18.0, 22.0));
    v /= 2.0;
    assert_eq!(v, Vec4::new(5.0, 7.0, 9.0, 11.0));
}

#[test]
#[should_panic]
fn test_index_out_of_bounds() {
    let v = Vec4::ZERO;
    let _ = v[4]; // This should panic
}
