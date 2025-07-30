// This file should be placed in your tests directory (e.g., `tests/vec2_tests.rs`)
// Make sure to add your math library as a dev-dependency in Cargo.toml
// and import the necessary types.

use crate::vec::{Vec2, Vec3};
use approx::AbsDiffEq;
use std::f32::consts::FRAC_PI_2; // Adjust this path

#[test]
fn test_construction_and_constants() {
    let v = Vec2::new(1.0, 2.0);
    assert_eq!(v.x, 1.0);
    assert_eq!(v.y, 2.0);

    assert_eq!(Vec2::ZERO, Vec2::new(0.0, 0.0));
    assert_eq!(Vec2::X, Vec2::new(1.0, 0.0));
    assert_eq!(Vec2::Y, Vec2::new(0.0, 1.0));
    assert!(Vec2::NAN.is_nan());
}

#[test]
fn test_conversion() {
    let v = Vec2::new(1.0, 2.0);
    assert_eq!(v.to_array(), [1.0, 2.0]);
    assert_eq!(Vec2::from_array([1.0, 2.0]), v);
    assert_eq!(v.to_tuple(), (1.0, 2.0));
    assert_eq!(Vec2::from_tuple((1.0, 2.0)), v);
    assert_eq!(v.extend(3.0), Vec3::new(1.0, 2.0, 3.0));
}

#[test]
fn test_math_utilities() {
    let v = Vec2::new(-1.0, 2.0);
    assert_eq!(v.abs(), Vec2::new(1.0, 2.0));
    assert_eq!(v.signum(), Vec2::new(-1.0, 1.0));
    assert_eq!(v.clamp(-0.5, 1.5), Vec2::new(-0.5, 1.5));
    assert_eq!(v.min(Vec2::new(-2.0, 3.0)), Vec2::new(-2.0, 2.0));
    // Corrected the expected result for the max function
    assert_eq!(v.max(Vec2::new(-2.0, 1.0)), Vec2::new(-1.0, 2.0));
}

#[test]
fn test_magnitude_and_normalization() {
    let v = Vec2::new(3.0, 4.0);
    assert!((v.length() - 5.0).abs() < 1e-6);
    assert_eq!(v.squared_length(), 25.0);

    let norm = v.normalize();
    assert!(norm.is_normalized());
    assert!(norm.abs_diff_eq(&Vec2::new(0.6, 0.8), 1e-6));

    assert!(v.try_normalize().is_some());
    assert!(Vec2::ZERO.try_normalize().is_none());
    assert_eq!(Vec2::ZERO.normalize_or_zero(), Vec2::ZERO);
}

#[test]
fn test_dot_cross_and_angle() {
    let v1 = Vec2::new(1.0, 0.0);
    let v2 = Vec2::new(0.0, 1.0);
    assert_eq!(v1.dot(v2), 0.0);
    assert_eq!(v1.cross(v2), 1.0);
    assert!((v1.angle_between_radians(v2) - FRAC_PI_2).abs() < 1e-6);
}

#[test]
fn test_interpolation() {
    let v1 = Vec2::new(0.0, 0.0);
    let v2 = Vec2::new(10.0, 10.0);
    assert_eq!(v1.lerp(v2, 0.5), Vec2::new(5.0, 5.0));
    assert_eq!(v1.lerp(v2, 1.5), Vec2::new(15.0, 15.0));
    assert_eq!(v1.lerp_clamped(v2, 1.5), v2);
}

#[test]
fn test_projection_and_reflection() {
    let v = Vec2::new(3.0, 4.0);
    let onto = Vec2::new(1.0, 0.0);
    assert!(v.project(onto).abs_diff_eq(&Vec2::new(3.0, 0.0), 1e-6));
    assert!(v.reject(onto).abs_diff_eq(&Vec2::new(0.0, 4.0), 1e-6));

    let incident = Vec2::new(1.0, -1.0);
    let normal = Vec2::new(0.0, 1.0);
    assert!(incident
        .reflect(normal)
        .abs_diff_eq(&Vec2::new(1.0, 1.0), 1e-6));
}

#[test]
fn test_distance_and_movement() {
    let v1 = Vec2::new(1.0, 2.0);
    let v2 = Vec2::new(4.0, 6.0);
    assert!((v1.distance(v2) - 5.0).abs() < 1e-6);
    assert_eq!(v1.squared_distance(v2), 25.0);

    let moved = Vec2::move_towards(v1, v2, 2.0);
    assert!(moved.abs_diff_eq(&Vec2::new(2.2, 3.6), 1e-6));

    let moved_fully = Vec2::move_towards(v1, v2, 10.0);
    assert_eq!(moved_fully, v2);
}

#[test]
fn test_operator_overloads() {
    let v1 = Vec2::new(1.0, 2.0);
    let v2 = Vec2::new(3.0, 4.0);

    assert_eq!(v1 + v2, Vec2::new(4.0, 6.0));
    assert_eq!(v1 - v2, Vec2::new(-2.0, -2.0));
    assert_eq!(v1 * v2, Vec2::new(3.0, 8.0));
    assert_eq!(v1 * 2.0, Vec2::new(2.0, 4.0));
    assert_eq!(v1 / 2.0, Vec2::new(0.5, 1.0));
    assert_eq!(-v1, Vec2::new(-1.0, -2.0));
}

#[test]
fn test_assignment_operators() {
    let mut v = Vec2::new(1.0, 2.0);
    v += Vec2::new(3.0, 4.0);
    assert_eq!(v, Vec2::new(4.0, 6.0));
    v -= Vec2::new(1.0, 1.0);
    assert_eq!(v, Vec2::new(3.0, 5.0));
    v *= 2.0;
    assert_eq!(v, Vec2::new(6.0, 10.0));
    v /= 2.0;
    assert_eq!(v, Vec2::new(3.0, 5.0));
}

#[test]
#[should_panic]
fn test_index_out_of_bounds() {
    let v = Vec2::ZERO;
    let _ = v[2]; // This should panic
}
