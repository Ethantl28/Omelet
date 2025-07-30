use crate::vec::{Vec2, Vec3, Vec4};
use approx::AbsDiffEq;
use std::f32::consts::FRAC_PI_2; // Adjust this path

#[test]
fn test_construction_and_constants() {
    let v = Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(v.x, 1.0);
    assert_eq!(v.y, 2.0);
    assert_eq!(v.z, 3.0);

    assert_eq!(Vec3::ZERO, Vec3::new(0.0, 0.0, 0.0));

    assert_eq!(Vec3::X, Vec3::new(1.0, 0.0, 0.0));
    assert_eq!(Vec3::Y, Vec3::new(0.0, 1.0, 0.0));
    assert_eq!(Vec3::Z, Vec3::new(0.0, 0.0, 1.0));
    assert!(Vec3::NAN.is_nan());
}

#[test]
fn test_conversion() {
    let v = Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(v.to_array(), [1.0, 2.0, 3.0]);
    assert_eq!(Vec3::from_array([1.0, 2.0, 3.0]), v);
    assert_eq!(v.to_tuple(), (1.0, 2.0, 3.0));
    assert_eq!(Vec3::from_tuple((1.0, 2.0, 3.0)), v);
    assert_eq!(Vec3::from_vec2_z(Vec2::new(1.0, 2.0), 3.0), v);
    assert_eq!(v.extend(4.0), Vec4::new(1.0, 2.0, 3.0, 4.0));
}

#[test]
fn test_math_utilities() {
    let v = Vec3::new(-1.0, 2.0, -3.0);
    assert_eq!(v.abs(), Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(v.signum(), Vec3::new(-1.0, 1.0, -1.0));
    assert_eq!(v.clamp(-0.5, 1.5), Vec3::new(-0.5, 1.5, -0.5));
    assert_eq!(
        v.min(Vec3::new(-2.0, 3.0, -2.0)),
        Vec3::new(-2.0, 2.0, -3.0)
    );
    assert_eq!(
        v.max(Vec3::new(-2.0, 1.0, -4.0)),
        Vec3::new(-1.0, 2.0, -3.0)
    );
}

#[test]
fn test_magnitude_and_normalization() {
    let v = Vec3::new(2.0, 3.0, 6.0); // Length is 7
    assert!((v.length() - 7.0).abs() < 1e-6);
    assert_eq!(v.squared_length(), 49.0);

    let norm = v.normalize();
    assert!(norm.is_normalized());
    assert!(norm.abs_diff_eq(&Vec3::new(2.0 / 7.0, 3.0 / 7.0, 6.0 / 7.0), 1e-6));

    assert!(v.try_normalize().is_some());
    assert!(Vec3::ZERO.try_normalize().is_none());
    assert_eq!(Vec3::ZERO.normalize_or_zero(), Vec3::ZERO);
}

#[test]
fn test_dot_cross_and_angle() {
    let v1 = Vec3::X;
    let v2 = Vec3::Y;
    assert_eq!(v1.dot(v2), 0.0);
    assert_eq!(v1.cross(v2), Vec3::Z);
    assert!((Vec3::angle_between_radians(v1, v2) - FRAC_PI_2).abs() < 1e-6);
}

#[test]
fn test_interpolation() {
    let v1 = Vec3::new(0.0, 0.0, 0.0);
    let v2 = Vec3::new(10.0, 10.0, -20.0);
    assert!(v1
        .lerp(v2, 0.5)
        .abs_diff_eq(&Vec3::new(5.0, 5.0, -10.0), 1e-6));
}

#[test]
fn test_projection_and_reflection() {
    let v = Vec3::new(3.0, 4.0, 5.0);
    let onto = Vec3::X;
    assert!(v.project(onto).abs_diff_eq(&Vec3::new(3.0, 0.0, 0.0), 1e-6));
    assert!(v.reject(onto).abs_diff_eq(&Vec3::new(0.0, 4.0, 5.0), 1e-6));

    let incident = Vec3::new(1.0, -1.0, 0.0);
    let normal = Vec3::Y;
    assert!(incident
        .reflect(normal)
        .abs_diff_eq(&Vec3::new(1.0, 1.0, 0.0), 1e-6));
}

#[test]
fn test_distance() {
    let v1 = Vec3::new(1.0, 2.0, 3.0);
    let v2 = Vec3::new(3.0, 4.0, 4.0);
    assert!((v1.distance(v2) - 3.0).abs() < 1e-6);
    assert_eq!(v1.squared_distance(v2), 9.0);
}

#[test]
fn test_random_vector() {
    let v = Vec3::random_unit_vector();
    assert!(v.is_normalized());
}

#[test]
fn test_operator_overloads() {
    let v1 = Vec3::new(1.0, 2.0, 3.0);
    let v2 = Vec3::new(4.0, 5.0, 6.0);

    assert_eq!(v1 + v2, Vec3::new(5.0, 7.0, 9.0));
    assert_eq!(v1 - v2, Vec3::new(-3.0, -3.0, -3.0));
    assert_eq!(v1 * v2, Vec3::new(4.0, 10.0, 18.0));
    assert_eq!(v1 * 2.0, Vec3::new(2.0, 4.0, 6.0));
    assert_eq!(v1 / 2.0, Vec3::new(0.5, 1.0, 1.5));
    assert_eq!(-v1, Vec3::new(-1.0, -2.0, -3.0));
}

#[test]
fn test_assignment_operators() {
    let mut v = Vec3::new(1.0, 2.0, 3.0);
    v += Vec3::new(4.0, 5.0, 6.0);
    assert_eq!(v, Vec3::new(5.0, 7.0, 9.0));
    v -= Vec3::new(1.0, 1.0, 1.0);
    assert_eq!(v, Vec3::new(4.0, 6.0, 8.0));
    v *= 2.0;
    assert_eq!(v, Vec3::new(8.0, 12.0, 16.0));
    v /= 2.0;
    assert_eq!(v, Vec3::new(4.0, 6.0, 8.0));
}

#[test]
#[should_panic]
fn test_index_out_of_bounds() {
    let v = Vec3::ZERO;
    let _ = v[3]; // This should panic
}
