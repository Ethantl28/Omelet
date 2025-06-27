use super::vec4::Vec4;
use super::vec3::Vec3;

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
    fn test_vec4_normalize_zero() {
        let v = Vec4::new(0.0, 0.0, 0.0, 0.0).normalize();
        assert_eq!(v, Vec4::new(0.0, 0.0, 0.0, 0.0));
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
        assert_eq!(v.w, 0.0);  // Stays a vector
        assert!(v.xyz().approx_eq(Vec3::new(1.0/3.0, 2.0/3.0, 2.0/3.0)));
    }

    #[test]
    fn test_vec4_point_normalization() {
        let p = Vec4::new(1.0, 2.0, 2.0, 1.0).normalize();
        assert!(p.is_normalized());
        assert!(p.w != 0.0);  // Remains a point
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
}