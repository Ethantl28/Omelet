use super::vec2::Vec2;

#[cfg(test)]
mod vec2_tests {
    use super::*;
    use crate::utils::{epsilon_eq, radians_to_degrees};

    #[test]
    fn test_new_and_components() {
        let v = Vec2::new(3.0, -5.0);
        assert_eq!(v.x, 3.0);
        assert_eq!(v.y, -5.0);
    }

    #[test]
    fn test_zero() {
        let z = Vec2::zero();
        assert_eq!(z, Vec2::new(0.0, 0.0));
    }

    #[test]
    fn test_to_from_array() {
        let v = Vec2::new(1.5, 2.5);
        let arr = v.to_array();
        assert_eq!(arr, [1.5, 2.5]);

        let v2 = Vec2::from_array(arr);
        assert_eq!(v2, v);
    }

    #[test]
    fn test_equality_and_display() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(1.0, 2.0);
        let c = Vec2::new(1.0, 3.0);

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(format!("{}", a), "Vec2(1, 2)");
    }

    #[test]
    fn test_indexing() {
        let mut v = Vec2::new(7.0, 8.0);
        assert_eq!(v[0], 7.0);
        assert_eq!(v[1], 8.0);
        v[0] = 1.0;
        v[1] = 2.0;
        assert_eq!(v, Vec2::new(1.0, 2.0));
    }

    #[test]
    #[should_panic(expected = "Vec2 index out of bounds")]
    fn test_index_out_of_bounds() {
        let v = Vec2::new(1.0, 2.0);
        let _ = v[2];
    }

    #[test]
    fn test_from_tuple_and_back() {
        let v: Vec2 = (3.0, 4.0).into();
        assert_eq!(v, Vec2::new(3.0, 4.0));
        let t: (f32, f32) = v.into();
        assert_eq!(t, (3.0, 4.0));
    }

    #[test]
    fn test_from_array_and_back() {
        let v: Vec2 = [6.0, 9.0].into();
        assert_eq!(v, Vec2::new(6.0, 9.0));
        let a: [f32; 2] = v.into();
        assert_eq!(a, [6.0, 9.0]);
    }

    #[test]
    fn test_default() {
        let v = Vec2::default();
        assert_eq!(v, Vec2::zero());
    }

    #[test]
    fn test_add_sub_scalar_mul_div() {
        let a = Vec2::new(2.0, 3.0);
        let b = Vec2::new(4.0, 1.0);

        assert_eq!(a + b, Vec2::new(6.0, 4.0));
        assert_eq!(a - b, Vec2::new(-2.0, 2.0));
        assert_eq!(a * 2.0, Vec2::new(4.0, 6.0));
        assert_eq!(2.0 * a, Vec2::new(4.0, 6.0));
        assert_eq!(a / 2.0, Vec2::new(1.0, 1.5));
        assert_eq!(-a, Vec2::new(-2.0, -3.0));
    }

    #[test]
    fn test_dot_product() {
        let a = Vec2::new(1.0, 3.0);
        let b = Vec2::new(4.0, -2.0);
        assert_eq!(a.dot(b), (1.0 * 4.0) + (3.0 * -2.0)); // 4 - 6 = -2.0
    }

    #[test]
    fn test_cross_product_2d() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        let result = a.cross(b); // 1*4 - 2*3 = 4 - 6 = -2
        assert_eq!(result, -2.0);
    }

    #[test]
    fn test_length_and_squared() {
        let v = Vec2::new(3.0, 4.0);
        assert_eq!(v.squared_length(), 25.0);
        assert!(epsilon_eq(v.length(), 5.0, 1e-6));
    }

    #[test]
    fn test_distance_and_squared() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(4.0, 6.0);
        assert_eq!(a.squared_distance(b), 25.0);
        assert!(epsilon_eq(a.distance(b), 5.0, 1e-6));
    }

    #[test]
    fn test_normalize_and_normalized_or_zero() {
        let v = Vec2::new(3.0, 4.0);
        let n = v.normalize();
        assert!(epsilon_eq(n.length(), 1.0, 1e-6));
        assert!(epsilon_eq(n.x, 0.6, 1e-6));
        assert!(epsilon_eq(n.y, 0.8, 1e-6));
    }

    #[test]
    #[should_panic(expected = "Cannot normalize zero-length vector")]
    fn test_normalize_zero() {
        let _ = Vec2::zero().normalize();
    }

    #[test]
    fn test_is_zero() {
        assert!(Vec2::new(0.0, 0.0).is_zero());
        assert!(!Vec2::new(1.0, 0.0).is_zero());
    }

    #[test]
    fn test_is_finite_and_nan() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(f32::INFINITY, 1.0);
        let c = Vec2::new(f32::NAN, 0.0);

        assert!(a.is_finite());
        assert!(!b.is_finite());
        assert!(c.is_nan());
        assert!(!a.is_nan());
    }

    #[test]
    fn test_project_onto() {
        let a = Vec2::new(3.0, 4.0);
        let b = Vec2::new(1.0, 0.0);
        let proj = a.project(b);
        assert_eq!(proj, Vec2::new(3.0, 0.0));

        let onto_zero = a.project(Vec2::zero());
        assert_eq!(onto_zero, Vec2::zero());
    }

    #[test]
    fn test_reflect() {
        let incoming = Vec2::new(1.0, -1.0);
        let normal = Vec2::new(0.0, 1.0); // Upward normal
        let reflected = incoming.reflect(normal);

        assert_eq!(reflected, Vec2::new(1.0, 1.0)); // Reflect over Y axis
    }

    #[test]
    fn test_clamp_length_max_min() {
        let v = Vec2::new(5.0, 0.0);
        let clamped = v.clamp(3.0, 3.0);
        assert!(clamped == Vec2::new(3.0, 3.0));
    }

    #[test]
    fn test_lerp() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(10.0, 10.0);
        let halfway = a.lerp(b, 0.5);
        assert_eq!(halfway, Vec2::new(5.0, 5.0));
    }

    #[test]
    fn test_slerp() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        let slerped = Vec2::slerp(a, b, 0.5);
        let expected = Vec2::new(0.70710677, 0.70710677); // ~ sqrt(2)/2

        assert!(epsilon_eq(slerped.x, expected.x, 1e-5));
        assert!(epsilon_eq(slerped.y, expected.y, 1e-5));
    }

    #[test]
    fn test_angle_between_and_signed() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);

        let angle = a.angle_between_radians(b);
        assert!(epsilon_eq(angle, std::f32::consts::FRAC_PI_2, 1e-6));
    }

    #[test]
    fn test_rotate_and_rotated() {
        let mut v = Vec2::new(1.0, 0.0);
        v.rotate(std::f32::consts::FRAC_PI_2);
        assert!(epsilon_eq(v.x, 0.0, 1e-6));
        assert!(epsilon_eq(v.y, 1.0, 1e-6));

        // Also test rotating a 45-degree vector by another 45 degrees = 90 degrees
        let mut v2 = Vec2::new(1.0, 1.0).normalize(); // 45-degree vector
        v2.rotate(std::f32::consts::FRAC_PI_4); // rotate by 45 degrees

        let expected = Vec2::new(0.0, 1.0);
        assert!(epsilon_eq(v2.x, expected.x, 1e-6));
        assert!(epsilon_eq(v2.y, expected.y, 1e-6));
    }

    #[test]
    fn test_barycentric() {
        let v1 = Vec2::new(0.0, 0.0);
        let v2 = Vec2::new(1.0, 0.0);
        let v3 = Vec2::new(0.0, 1.0);

        // barycentric coords (1,0,0) should return v1
        let bary1 = Vec2::barycentric_simplified(v1, v2, v3, 1.0, 0.0, 0.0);
        assert_eq!(bary1, v1);

        // barycentric coords (0,1,0) should return v2
        let bary2 = Vec2::barycentric_simplified(v1, v2, v3, 0.0, 1.0, 0.0);
        assert_eq!(bary2, v2);

        // barycentric coords (0,0,1) should return v3
        let bary3 = Vec2::barycentric_simplified(v1, v2, v3, 0.0, 0.0, 1.0);
        assert_eq!(bary3, v3);

        // barycentric coords (0.25,0.25,0.5) should be the weighted average
        let bary4 = Vec2::barycentric_simplified(v1, v2, v3, 0.25, 0.25, 0.5);
        let expected = Vec2::new(
            0.25 * v1.x + 0.25 * v2.x + 0.5 * v3.x,
            0.25 * v1.y + 0.25 * v2.y + 0.5 * v3.y,
        );
        assert_eq!(bary4, expected);
    }

    #[test]
    fn test_component_min_max() {
        let a = Vec2::new(1.0, 3.0);
        let b = Vec2::new(2.0, 2.0);
        let min = a.min(b);
        let max = a.max(b);
        assert_eq!(min, Vec2::new(1.0, 2.0));
        assert_eq!(max, Vec2::new(2.0, 3.0));
    }

    #[test]
    fn test_abs() {
        let v = Vec2::new(-3.0, 4.0);
        let abs_v = v.abs();
        assert_eq!(abs_v, Vec2::new(3.0, 4.0));
    }
}
