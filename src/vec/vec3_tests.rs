use super::vec3::Vec3;

mod tests {
    use super::*;
    use core::f32::consts::PI;
    #[test]
    fn test_vec3_new_zero() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);

        let z = Vec3::zero();
        assert_eq!(z, Vec3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_vec3_is_zero() {
        assert!(Vec3::zero().is_zero());
        assert!(!Vec3::new(1.0, 0.0, 0.0).is_zero());
    }

    #[test]
    fn test_vec3_abs_signum_clamp() {
        let v = Vec3::new(-1.5, 2.0, -3.0);
        assert_eq!(v.abs(), Vec3::new(1.5, 2.0, 3.0));
        assert_eq!(v.signum(), Vec3::new(-1.0, 1.0, -1.0));
        assert_eq!(v.clamp(-1.0, 1.0), Vec3::new(-1.0, 1.0, -1.0));
    }

    #[test]
    fn test_vec3_array_conversion() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.to_array(), [1.0, 2.0, 3.0]);
        let arr: [f32; 3] = [4.0, 5.0, 6.0];
        assert_eq!(Vec3::from_array(arr), Vec3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn test_vec3_length_dot_cross() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert_eq!(v.length(), 5.0);
        assert_eq!(v.squared_length(), 25.0);

        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(a.dot(b), 0.0);
        assert_eq!(a.cross(b), Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_vec3_normalize_variants() {
        let v = Vec3::new(3.0, 0.0, 4.0);
        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < 1e-6);
        assert!(v.try_normalize().length() - 1.0 < 1e-6);
        assert!(Vec3::zero().try_normalize().is_zero());
    }

    #[test]
    fn test_vec3_direction_distance() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 6.0, 3.0);
        assert!((a.distance(b) - 5.0).abs() < 1e-6);
        assert_eq!(a.squared_distance(b), 25.0);
        assert!((a.direction_to(b).length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec3_project_reject_reflect() {
        let a = Vec3::new(2.0, 3.0, 4.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let proj = a.project(b);
        assert_eq!(proj, Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(a.reject(b), a - proj);
        let reflected = a.reflect(Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(reflected, Vec3::new(2.0, -3.0, 4.0));
    }

    #[test]
    fn test_vec3_min_max_lerp() {
        let a = Vec3::new(1.0, 4.0, 2.0);
        let b = Vec3::new(3.0, 2.0, 5.0);
        assert_eq!(a.min(b), Vec3::new(1.0, 2.0, 2.0));
        assert_eq!(a.max(b), Vec3::new(3.0, 4.0, 5.0));
        assert_eq!(Vec3::lerp(a, b, 0.5), Vec3::new(2.0, 3.0, 3.5));
    }

    #[test]
    fn test_vec3_angle_between() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert!((Vec3::angle_between_radians(a, b) - PI / 2.0).abs() < 1e-6);
        assert!((Vec3::angle_between_degrees(a, b) - 90.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec3_slerp_variants() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let s = Vec3::slerp(a, b, 0.5);
        assert!((s.length() - 1.0).abs() < 1e-6);
        let sa = Vec3::slerp_angle(a, b, 0.5);
        assert!((sa.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec3_barycentric_in_triangle() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        let p = Vec3::new(0.25, 0.25, 0.0);
        let (u, v, w) = Vec3::barycentric(p, a, b, c);
        assert!((u + v + w - 1.0).abs() < 1e-6);
        assert!(Vec3::in_triangle(p, a, b, c));
    }

    #[test]
    fn test_vec3_misc_utilities() {
        let v = Vec3::new(0.0, 0.0, 0.0);
        assert!(v.is_finite());
        assert!(!v.is_nan());
        let w = Vec3::new(f32::NAN, 1.0, 2.0);
        assert!(w.is_nan());
        let z = Vec3::new(1e-9, 0.0, 0.0);
        assert!(z.is_zero_eps(1e-6));
        assert_eq!(v.normalize_or_zero(), Vec3::zero());
        let m = Vec3::new(1.0, -2.0, 3.0);
        assert_eq!(m.mirror(Vec3::new(0.0, 1.0, 0.0)), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_vec3_triple_product_scalar() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let c = Vec3::new(0.0, 0.0, 1.0);

        // scalar triple product a · (b × c) = 1.0
        let result = Vec3::triple_product_scalar(a, b, c);
        assert!(crate::utils::epsilon_eq(result, 1.0, 1e-6));
    }

    #[test]
    fn test_vec3_triple_product_vector() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let c = Vec3::new(7.0, 8.0, 9.0);

        let result = Vec3::triple_product_vector(a, b, c);

        // vector triple product: b * (a · c) - a * (b · c)
        let ac = a.dot(c);
        let bc = b.dot(c);
        let expected = b * ac - a * bc;

        assert!(result.approx_eq(expected));
    }

    #[test]
    fn test_vec3_orthonormal_basis() {
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let (t, b) = Vec3::orthonormal_basis(&normal);
        assert!((t.length() - 1.0).abs() < 1e-6);
        assert!((b.length() - 1.0).abs() < 1e-6);
        assert!(t.dot(b).abs() < 1e-6);
        assert!(t.dot(normal).abs() < 1e-6);
        assert!(b.dot(normal).abs() < 1e-6);
    }

    #[test]
    fn test_vec3_mirror() {
        let v = Vec3::new(1.0, -2.0, 3.0);
        let axis = Vec3::new(0.0, 1.0, 0.0);
        let mirrored = v.mirror(axis);
        assert_eq!(mirrored, Vec3::new(1.0, 2.0, 3.0));

        let diag = Vec3::new(1.0, 1.0, 0.0).normalize();
        let v = Vec3::new(2.0, 0.0, 0.0);
        let mirrored = v.mirror(diag);
        let expected = v - diag * (2.0 * v.dot(diag));
        assert!((mirrored - expected).length() < 1e-6);
    }

    #[test]
    fn test_vec3_approx_eq_eps() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(1.0 + 1e-7, 2.0 - 1e-7, 3.0 + 1e-7);
        assert!(a.approx_eq_eps(b, 1e-6));
        assert!(!a.approx_eq_eps(b, 1e-9));
    }

    #[test]
    fn test_vec3_rotate_around_axis() {
        let v = Vec3::new(1.0, 0.0, 0.0);
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let rotated = v.rotate_around_axis(axis, PI / 2.0);
        assert!((rotated - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6);

        let axis = Vec3::new(1.0, 0.0, 0.0);
        let rotated = Vec3::new(0.0, 1.0, 0.0).rotate_around_axis(axis, PI);
        assert!((rotated - Vec3::new(0.0, -1.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn test_vec3_random_unit_vector() {
        for _ in 0..1000 {
            let v = Vec3::random_unit_vector();
            let len = v.length();
            assert!((len - 1.0).abs() < 1e-4, "length was {}", len);
        }
    }
}
