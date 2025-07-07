use super::vec3::Vec3;
use super::vec4::Vec4;

#[cfg(test)]
mod tests {
    use super::*;

    // Basic construction and properties
    #[test]
    fn test_vec4_construction() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
        assert_eq!(v.w, 4.0);
    }

    #[test]
    fn test_vec4_default() {
        let v = Vec4::new(0.0, 0.0, 0.0, 0.0);
        assert_eq!(v.length(), 0.0);
        assert_eq!(v.squared_length(), 0.0);
    }

    // Vector operations
    #[test]
    fn test_vec4_length() {
        let v = Vec4::new(1.0, 2.0, 2.0, 4.0);
        assert_eq!(v.length(), 5.0); // sqrt(1 + 4 + 4 + 16) = 5
    }

    #[test]
    fn test_vec4_squared_length() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.squared_length(), 30.0); // 1 + 4 + 9 + 16
    }

    #[test]
    fn test_vec4_normalize() {
        let v = Vec4::new(0.0, 3.0, 0.0, 4.0).normalize();
        assert!(v.is_normalized());
        assert!(v.is_normalized_fast());
        assert!(v.approx_eq(Vec4::new(0.0, 0.6, 0.0, 0.8)));
    }

    #[test]
    fn test_vec4_normalize_or_zero_zero() {
        let v = Vec4::new(0.0, 0.0, 0.0, 0.0).normalize_or_zero();
        assert_eq!(v, Vec4::new(0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    #[should_panic(expected = "Cannot normalize zero length vector")]
    fn test_vec4_normalize_should_panic_on_zero() {
        let _ = Vec4::new(0.0, 0.0, 0.0, 0.0).normalize();
    }

    #[test]
    fn test_vec4_try_normalize_zero() {
        let v = Vec4::new(0.0, 0.0, 0.0, 0.0);
        assert_eq!(v.try_normalize(), None);
    }

    // Dot product
    #[test]
    fn test_vec4_dot() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(a.dot(b), 70.0); // 5 + 12 + 21 + 32
    }

    // Cross product (XYZ only)
    #[test]
    fn test_vec4_cross_xyz() {
        let a = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let b = Vec4::new(0.0, 1.0, 0.0, 0.0);
        assert_eq!(a.cross_xyz(b), Vec4::new(0.0, 0.0, 1.0, 0.0));
    }

    #[test]
    fn test_vec4_cross_xyz_anticommutative() {
        let a = Vec4::new(1.0, 2.0, 3.0, 0.0);
        let b = Vec4::new(4.0, 5.0, 6.0, 0.0);
        assert_eq!(a.cross_xyz(b), -b.cross_xyz(a));
    }

    // Distance operations
    #[test]
    fn test_vec4_distance() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(a.distance(b), 8.0); // sqrt(16*4)
    }

    #[test]
    fn test_vec4_squared_distance() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(a.squared_distance(b), 64.0); // 16*4
    }

    // Direction operations
    #[test]
    fn test_vec4_direction_to() {
        let a = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let b = Vec4::new(2.0, 0.0, 0.0, 0.0);
        assert_eq!(a.direction_to(b), Vec4::new(1.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_vec4_direction_to_normalized() {
        let a = Vec4::new(1.0, 1.0, 0.0, 0.0);
        let b = Vec4::new(2.0, 2.0, 0.0, 0.0);
        let dir = a.direction_to(b);
        assert!(dir.is_normalized());
    }

    // Projection and rejection
    #[test]
    fn test_vec4_project() {
        let a = Vec4::new(1.0, 1.0, 0.0, 0.0);
        let b = Vec4::new(1.0, 0.0, 0.0, 0.0);
        assert_eq!(a.project(b), Vec4::new(1.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_vec4_reject() {
        let a = Vec4::new(1.0, 1.0, 0.0, 0.0);
        let b = Vec4::new(1.0, 0.0, 0.0, 0.0);
        assert_eq!(a.reject(b), Vec4::new(0.0, 1.0, 0.0, 0.0));
    }

    // Reflection
    #[test]
    fn test_vec4_reflect() {
        let v = Vec4::new(1.0, -1.0, 0.0, 0.0);
        let normal = Vec4::new(0.0, 1.0, 0.0, 0.0);
        assert_eq!(v.reflect(normal), Vec4::new(1.0, 1.0, 0.0, 0.0));
    }

    // Interpolation
    #[test]
    fn test_vec4_lerp() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(Vec4::lerp(a, b, 0.5), Vec4::new(3.0, 4.0, 5.0, 6.0));
    }

    #[test]
    fn test_vec4_slerp() {
        let a = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let b = Vec4::new(0.0, 1.0, 0.0, 0.0);
        let result = Vec4::slerp(a, b, 0.5);
        let expected = Vec4::new(0.7071068, 0.7071068, 0.0, 0.0);
        assert!(result.approx_eq(expected));
    }

    // Operator overloading
    #[test]
    fn test_vec4_add() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(a + b, Vec4::new(6.0, 8.0, 10.0, 12.0));
    }

    #[test]
    fn test_vec4_sub() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(a - b, Vec4::new(-4.0, -4.0, -4.0, -4.0));
    }

    #[test]
    fn test_vec4_mul() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(a * b, Vec4::new(5.0, 12.0, 21.0, 32.0));
    }

    #[test]
    fn test_vec4_scalar_mul() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(a * 2.0, Vec4::new(2.0, 4.0, 6.0, 8.0));
        assert_eq!(2.0 * a, Vec4::new(2.0, 4.0, 6.0, 8.0));
    }

    #[test]
    fn test_vec4_div() {
        let a = Vec4::new(4.0, 9.0, 16.0, 25.0);
        let b = Vec4::new(2.0, 3.0, 4.0, 5.0);
        assert_eq!(a / b, Vec4::new(2.0, 3.0, 4.0, 5.0));
    }

    #[test]
    fn test_vec4_scalar_div() {
        let a = Vec4::new(2.0, 4.0, 6.0, 8.0);
        assert_eq!(a / 2.0, Vec4::new(1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_vec4_neg() {
        let a = Vec4::new(1.0, -2.0, 3.0, -4.0);
        assert_eq!(-a, Vec4::new(-1.0, 2.0, -3.0, 4.0));
    }

    // Edge cases
    #[test]
    fn test_vec4_move_towards_exact() {
        let current = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let target = Vec4::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(Vec4::move_towards(current, target, 8.0), target);
    }

    #[test]
    fn test_vec4_move_towards_partial() {
        let current = Vec4::new(0.0, 0.0, 0.0, 0.0);
        let target = Vec4::new(3.0, 4.0, 0.0, 0.0);
        let result = Vec4::move_towards(current, target, 2.5);
        assert_eq!(result, Vec4::new(1.5, 2.0, 0.0, 0.0));
    }

    #[test]
    fn test_vec4_approx_eq() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(1.000001, 2.000001, 3.000001, 4.000001);
        assert!(a.approx_eq(b));
    }

    // Homogeneous coordinate tests
    #[test]
    fn test_vec4_xyz() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.xyz(), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_vec4_vector_normalization() {
        let v = Vec4::new(1.0, 2.0, 2.0, 0.0).normalize();
        assert!(v.is_normalized());
        assert_eq!(v.w, 0.0); // Stays a vector
        assert!(
            v.xyz()
                .approx_eq(Vec3::new(1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0))
        );
    }

    #[test]
    fn test_vec4_point_normalization() {
        let p = Vec4::new(1.0, 2.0, 2.0, 1.0).normalize();
        assert!(p.is_normalized());
        assert!(p.w != 0.0); // Remains a point
    }

    #[test]
    fn test_vec4_point_vector_behavior() {
        let point = Vec4::new(1.0, 2.0, 3.0, 1.0);
        let vector = Vec4::new(1.0, 2.0, 3.0, 0.0);

        // Vectors should normalize XYZ components only
        let norm_vec = vector.normalize();
        assert_eq!(norm_vec.w, 0.0);

        // Points should normalize all components
        let norm_point = point.normalize();
        assert!(norm_point.w != 0.0);
    }

    #[test]
    fn test_vec4_zero_nan_infinity() {
        assert_eq!(Vec4::zero(), Vec4::new(0.0, 0.0, 0.0, 0.0));
        assert!(Vec4::nan().is_nan());
        assert!(!Vec4::infinity().is_nan());
        assert!(Vec4::infinity().is_finite() == false);
    }

    #[test]
    fn test_vec4_from_vec3_w() {
        let v3 = Vec3::new(1.0, 2.0, 3.0);
        let v4 = Vec4::from_vec3_w(v3, 4.0);
        assert_eq!(v4, Vec4::new(1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_vec4_signum_abs_clamp_min_max() {
        let v = Vec4::new(-1.5, 2.0, -3.0, 0.5);
        assert_eq!(v.abs(), Vec4::new(1.5, 2.0, 3.0, 0.5));
        assert_eq!(v.signum(), Vec4::new(-1.0, 1.0, -1.0, 1.0));
        assert_eq!(v.clamp(-1.0, 1.0), Vec4::new(-1.0, 1.0, -1.0, 0.5));
        let b = Vec4::new(0.0, 3.0, -4.0, 2.0);
        assert_eq!(v.min(b), Vec4::new(-1.5, 2.0, -4.0, 0.5));
        assert_eq!(v.max(b), Vec4::new(0.0, 3.0, -3.0, 2.0));
    }

    #[test]
    fn test_vec4_perpendicular() {
        let v = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let p = v.perpendicular();
        assert!(crate::utils::epsilon_eq_default(v.dot(p), 0.0));
        assert!(p.length() > 0.0);
    }

    #[test]
    fn test_vec4_rotate_in_plane() {
        let v = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let rotated = v.rotate_in_plane(0, 1, std::f32::consts::FRAC_PI_2);
        assert!(rotated.approx_eq(Vec4::new(0.0, 1.0, 0.0, 0.0)));
    }

    #[test]
    fn test_vec4_rotate_around_axis() {
        let axis = Vec4::new(0.0, 0.0, 1.0, 0.0);
        let v = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let rotated = v.rotate_around_axis(axis, std::f32::consts::FRAC_PI_2);
        assert!(rotated.approx_eq(Vec4::new(0.0, 1.0, 0.0, 0.0)));
    }

    #[test]
    fn test_vec4_random_unit_vector() {
        for _ in 0..100 {
            let v = Vec4::random_unit_vector();
            assert!(v.is_finite());
            assert!((v.length() - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_vec4_random_in_unit_sphere() {
        for _ in 0..100 {
            let v = Vec4::random_in_unit_sphere();
            assert!(v.length() <= 1.0);
        }
    }

    #[test]
    fn test_vec4_hypervolume_4d() {
        let a = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let b = Vec4::new(0.0, 1.0, 0.0, 0.0);
        let c = Vec4::new(0.0, 0.0, 1.0, 0.0);
        let d = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let vol = Vec4::hypervolume_4d(a, b, c, d);
        assert!(crate::utils::epsilon_eq_default(vol, 1.0));
    }

    #[test]
    fn test_vec4_barycentric_in_triangle() {
        let a = Vec4::new(0.0, 0.0, 0.0, 0.0);
        let b = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let c = Vec4::new(0.0, 1.0, 0.0, 0.0);
        let p = Vec4::new(0.25, 0.25, 0.0, 0.0);
        let (u, v, w) = Vec4::barycentric(p, a, b, c);
        assert!(crate::utils::epsilon_eq_default(u + v + w, 1.0));
        assert!(Vec4::in_triangle(p, a, b, c));
    }

    #[test]
    fn test_vec4_is_zero_and_finiteness() {
        let z = Vec4::zero();
        assert!(z.is_zero());
        assert!(z.is_zero_eps(1e-6));
        assert!(z.is_finite());
        assert!(!z.is_nan());

        let n = Vec4::nan();
        assert!(n.is_nan());

        let i = Vec4::infinity();
        assert!(!i.is_finite());
    }
}
