use super::vec3::Vec3;

mod tests {
    use crate::vec::Vec2;

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

        let try_n = v.try_normalize();
        assert!(try_n.is_some());
        assert!((try_n.unwrap().length() - 1.0).abs() < 1e-6);

        let zero_vector = Vec3::zero();
        let try_z = zero_vector.try_normalize();
        assert!(try_z.is_none());

        let or_zero = zero_vector.normalize_or_zero();
        assert!(or_zero.is_zero());
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

    #[test]
    fn test_vec3_tuple_conversion() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let t: (f32, f32, f32) = v.into();
        assert_eq!(t, (1.0, 2.0, 3.0));

        let v2: Vec3 = (4.0, 5.0, 6.0).into();
        assert_eq!(v2, Vec3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn test_vec3_indexing() {
        let v = Vec3::new(10.0, 20.0, 30.0);
        assert_eq!(v[0], 10.0);
        assert_eq!(v[1], 20.0);
        assert_eq!(v[2], 30.0);
    }

    #[test]
    fn test_vec3_display() {
        let v = Vec3::new(1.1, 2.2, 3.3);
        assert_eq!(format!("{}", v), "Vec3(1.1, 2.2, 3.3)");
    }

    #[test]
    fn test_vec3_from_vec2_z() {
        let v2 = Vec2::new(1.0, 2.0);
        let v3 = Vec3::from_vec2_z(v2, 3.0);
        assert_eq!(v3, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_vec3_nan_infinity() {
        let n = Vec3::nan();
        assert!(n.x.is_nan() && n.y.is_nan() && n.z.is_nan());

        let i = Vec3::infinity();
        assert!(i.x.is_infinite() && i.y.is_infinite() && i.z.is_infinite());
    }

    #[test]
    fn test_vec3_angle_to() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let angle = a.angle_to(b);
        assert!((angle - PI / 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec3_lerp_clamped_variants() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 6.0, 8.0);

        let unclamped = a.lerp_clamped(b, 1.5);
        assert_eq!(unclamped, b);

        let between = Vec3::lerp_between(a, b, 0.5);
        assert_eq!(between, Vec3::new(2.5, 4.0, 5.5));

        let clamped = Vec3::lerp_between_clamped(a, b, -1.0);
        assert_eq!(clamped, a);
    }

    #[test]
    fn test_vec3_move_towards() {
        let start = Vec3::new(0.0, 0.0, 0.0);
        let target = Vec3::new(10.0, 0.0, 0.0);
        let moved = Vec3::move_towards(start, target, 3.0);
        assert_eq!(moved, Vec3::new(3.0, 0.0, 0.0));

        let overshoot = Vec3::move_towards(start, target, 20.0);
        assert_eq!(overshoot, target);
    }

    #[test]
    fn test_vec3_orthonormal_basis_from_vector() {
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let (v, w) = normal.orthonormal_basis();

        // Both should be normalized
        assert!((v.length() - 1.0).abs() < 1e-6);
        assert!((w.length() - 1.0).abs() < 1e-6);

        // Both should be perpendicular to the original vector
        assert!(v.dot(normal).abs() < 1e-6);
        assert!(w.dot(normal).abs() < 1e-6);

        // v and w should also be orthogonal to each other
        assert!(v.dot(w).abs() < 1e-6);
    }

    #[test]
    fn test_vec3_orthonormalize() {
        use crate::utils::are_orthonormal;

        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 1.0, 0.0);

        let (a_ortho, b_ortho) = Vec3::orthonormalize(a, b);
        let c_ortho = a_ortho.cross(b_ortho);

        assert!(
            are_orthonormal(a_ortho, b_ortho, c_ortho, 1e-6),
            "Vectors are not orthonormal"
        );
    }
}
