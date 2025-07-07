use super::vec2::Vec2;

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    #[test]
    fn test_new_and_zero() {
        let v = Vec2::new(1.0, -2.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, -2.0);

        let z = Vec2::zero();
        assert_eq!(z, Vec2::new(0.0, 0.0));
    }

    #[test]
    fn test_abs_signum_clamp() {
        let v = Vec2::new(-3.0, 4.0);
        assert_eq!(v.abs(), Vec2::new(3.0, 4.0));
        assert_eq!(v.signum(), Vec2::new(-1.0, 1.0));
        assert_eq!(v.clamp(-2.0, 2.0), Vec2::new(-2.0, 2.0));
    }

    #[test]
    fn test_length_and_squared_length() {
        let v = Vec2::new(3.0, 4.0);
        assert_eq!(v.length(), 5.0);
        assert_eq!(v.squared_length(), 25.0);
    }

    #[test]
    fn test_normalization() {
        let v = Vec2::new(3.0, 4.0);
        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_cross_perpendicular() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        assert_eq!(a.dot(b), 0.0);
        assert_eq!(a.cross(b), 1.0);
        assert_eq!(a.perpendicular(), b);
    }

    #[test]
    fn test_projection_rejection_reflection() {
        let a = Vec2::new(3.0, 4.0);
        let b = Vec2::new(1.0, 0.0);
        assert_eq!(a.project(b), Vec2::new(3.0, 0.0));
        assert_eq!(a.reject(b), Vec2::new(0.0, 4.0));
        assert_eq!(a.reflect(Vec2::new(0.0, 1.0)), Vec2::new(3.0, -4.0));
    }

    #[test]
    fn test_distance_and_direction() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(3.0, 4.0);
        assert_eq!(a.distance(b), 5.0);
        assert_eq!(a.squared_distance(b), 25.0);
        assert_eq!(a.direction_to_raw(b), b);
        assert!((a.direction_to(b).length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_angles() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        assert!((a.angle_between_radians(b) - PI / 2.0).abs() < 1e-6);
        assert!((a.angle_between_degrees(b) - 90.0).abs() < 1e-6);
        assert!((a.angle_to_radians(b) - PI / 2.0).abs() < 1e-6);
        assert!((a.angle_to_degrees(b) - 90.0).abs() < 1e-6);
    }

    #[test]
    fn test_lerp_slerp() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(10.0, 0.0);
        assert_eq!(Vec2::lerp(&a, b, 0.5), Vec2::new(5.0, 0.0));
        assert_eq!(Vec2::lerp_clamped(&a, b, 2.0), b);

        let c = Vec2::new(1.0, 0.0);
        let d = Vec2::new(0.0, 1.0);
        let s = Vec2::slerp(c, d, 0.5);
        assert!((s.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_barycentric_and_in_triangle() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(1.0, 0.0);
        let c = Vec2::new(0.0, 1.0);
        let p = Vec2::new(0.3, 0.3);
        let (u, v, w) = Vec2::barycentric(p, a, b, c);
        assert!((u + v + w - 1.0).abs() < 1e-6);
        assert!(Vec2::in_triangle(p, a, b, c));
    }

    #[test]
    fn test_utilities_and_checks() {
        let v = Vec2::new(1.0, f32::INFINITY);
        assert!(!v.is_finite());
        assert!(!Vec2::new(f32::NAN, 0.0).is_finite());
        assert!(Vec2::new(f32::NAN, 0.0).is_nan());
        assert!(Vec2::zero().is_zero());
    }

    #[test]
    fn test_rotation_and_from_angle() {
        let mut v = Vec2::new(1.0, 0.0);
        v.rotate(PI / 2.0);
        assert!((v - Vec2::new(0.0, 1.0)).length() < 1e-6);

        let around = v.rotate_around(Vec2::new(1.0, 1.0), PI);
        assert!((around - Vec2::new(2.0, 1.0)).length() < 1e-6);

        let angle = Vec2::from_angle(PI);
        assert!((angle + Vec2::new(1.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn test_operator_overloads() {
        let a = Vec2::new(2.0, 3.0);
        let b = Vec2::new(1.0, 2.0);

        assert_eq!(a + b, Vec2::new(3.0, 5.0));
        assert_eq!(a - b, Vec2::new(1.0, 1.0));
        assert_eq!(a * b, Vec2::new(2.0, 6.0));
        assert_eq!(a / b, Vec2::new(2.0, 1.5));

        assert_eq!(a + 1.0, Vec2::new(3.0, 4.0));
        assert_eq!(1.0 + a, Vec2::new(3.0, 4.0));
        assert_eq!(a * 2.0, Vec2::new(4.0, 6.0));
        assert_eq!(2.0 * a, Vec2::new(4.0, 6.0));

        assert_eq!(-a, Vec2::new(-2.0, -3.0));

        assert!(a == Vec2::new(2.0, 3.0));
    }

    #[test]
    fn test_indexing_and_conversion() {
        let v = Vec2::new(4.0, 5.0);
        assert_eq!(v[0], 4.0);
        assert_eq!(v[1], 5.0);

        let mut m = v;
        m[0] = 7.0;
        assert_eq!(m.x, 7.0);

        let tup: (f32, f32) = v.into();
        assert_eq!(tup, (4.0, 5.0));

        let arr: [f32; 2] = v.into();
        assert_eq!(arr, [4.0, 5.0]);

        let v2: Vec2 = [1.0, 2.0].into();
        assert_eq!(v2, Vec2::new(1.0, 2.0));
    }

    #[test]
    fn test_display() {
        let v = Vec2::new(1.23, 4.56);
        assert_eq!(format!("{}", v), "Vec2(1.23, 4.56)");
    }

    #[test]
    fn test_vec2_mirror() {
        let v = Vec2::new(1.0, -2.0);
        let axis = Vec2::new(0.0, 1.0);
        let mirrored = v.mirror(axis);
        assert_eq!(mirrored, Vec2::new(1.0, 2.0));

        let axis_diag = Vec2::new(1.0, 1.0).normalize();
        let v = Vec2::new(2.0, 0.0);
        let mirrored = v.mirror(axis_diag);
        let expected = v - axis_diag * (2.0 * v.dot(axis_diag));
        assert!((mirrored - expected).length() < 1e-6);
    }

    #[test]
    fn test_vec2_normal() {
        let v = Vec2::new(2.0, 0.0);
        let n = v.normal();
        assert_eq!(n, Vec2::new(0.0, -2.0));
        assert!(n.dot(v).abs() < 1e-6);

        let z = Vec2::zero();
        assert_eq!(z.normal(), Vec2::zero());
    }

    #[test]
    fn test_vec2_normalize_or_zero() {
        let zero = Vec2::zero();
        assert_eq!(zero.normalize_or_zero(), Vec2::zero());

        let v = Vec2::new(3.0, 4.0);
        let n = v.normalize_or_zero();
        assert!((n.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_nan_and_infinity() {
        let v_nan = Vec2::nan();
        assert!(v_nan.is_nan());

        let v_inf = Vec2::infinity();
        assert!(!v_inf.is_nan());
        assert!(!v_inf.is_finite());
    }

    #[test]
    fn test_min_and_max() {
        let a = Vec2::new(1.0, 5.0);
        let b = Vec2::new(2.0, 4.0);

        assert_eq!(a.min(b), Vec2::new(1.0, 4.0));
        assert_eq!(a.max(b), Vec2::new(2.0, 5.0));
    }

    #[test]
    fn test_try_normalize() {
        let v = Vec2::new(3.0, 4.0);
        let normalized = v.try_normalize().unwrap();
        assert!((normalized.length() - 1.0).abs() < 1e-6);

        let zero = Vec2::zero();
        assert!(zero.try_normalize().is_none());
    }

    #[test]
    fn test_is_zero_eps() {
        let v = Vec2::new(1e-4, 1e-4);
        assert!(v.is_zero_eps(1e-3));
        assert!(!v.is_zero_eps(1e-5));
    }

    #[test]
    fn test_is_normalized_and_fast() {
        let v = Vec2::new(3.0, 4.0).normalize();
        assert!(v.is_normalized());
        assert!(v.is_normalized_fast());

        let v = Vec2::new(10.0, 0.0);
        assert!(!v.is_normalized());
        assert!(!v.is_normalized_fast());
    }

    #[test]
    fn test_random_unit_vector() {
        for _ in 0..100 {
            let v = Vec2::random_unit_vector();
            assert!((v.length() - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn test_move_towards() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(3.0, 4.0);

        let result = Vec2::move_towards(a, b, 5.0);
        assert_eq!(result, b); // moves entire way

        let result = Vec2::move_towards(a, b, 2.0);
        assert!((result.length() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_barycentric_simplified() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(1.0, 0.0);
        let c = Vec2::new(0.0, 1.0);

        let result = Vec2::barycentric_simplified(a, b, c, 0.25, 0.25, 0.5);
        assert_eq!(result, Vec2::new(0.25, 0.5));
    }

    #[test]
    fn test_lerp_variants() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(10.0, 0.0);

        let result = a.lerp(b, 0.5);
        assert_eq!(result, Vec2::new(5.0, 0.0));

        let result = a.lerp_clamped(b, 2.0);
        assert_eq!(result, b); // t > 1.0 gets clamped

        let result = a.lerp_clamped(b, -1.0);
        assert_eq!(result, a); // t < 0.0 gets clamped

        let result = Vec2::lerp_between(a, b, 0.25);
        assert_eq!(result, Vec2::new(2.5, 0.0));

        let result = Vec2::lerp_between_clamped(a, b, 1.5);
        assert_eq!(result, b); // t clamped to 1.0

        let result = Vec2::lerp_between_clamped(a, b, -0.5);
        assert_eq!(result, a); // t clamped to 0.0
    }
}
