use super::vec3::Vec3;

///Vec3 tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::epsilon_eq;

    // Basic construction and properties
    #[test]
    fn test_vec3_construction() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn test_vec3_default() {
        let v = Vec3::new(0.0, 0.0, 0.0);
        assert_eq!(v.length(), 0.0);
        assert_eq!(v.squared_length(), 0.0);
    }

    // Vector operations
    #[test]
    fn test_vec3_length() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert!(epsilon_eq(v.length(), (14.0f32).sqrt(), 1e-6));
    }

    #[test]
    fn test_vec3_squared_length() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.squared_length(), 14.0);
    }

    #[test]
    fn test_vec3_normalize() {
        let v = Vec3::new(1.0, 2.0, 3.0).normalize();
        assert!(v.is_normalized());
        assert!(v.is_normalized_fast());
    }

    #[test]
    fn test_vec3_normalize_zero() {
        let v = Vec3::new(0.0, 0.0, 0.0).normalize();
        assert_eq!(v, Vec3::new(0.0, 0.0, 0.0));
    }

    // Dot and cross products
    #[test]
    fn test_vec3_dot() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.dot(b), 32.0);
    }

    #[test]
    fn test_vec3_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(a.cross(b), Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_vec3_cross_anticommutative() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.cross(b), -b.cross(a));
    }

    // Distance operations
    #[test]
    fn test_vec3_distance() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 6.0, 8.0);
        assert!(epsilon_eq(a.distance(b), (50.0f32).sqrt(), 1e-6));
    }

    #[test]
    fn test_vec3_squared_distance() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 6.0, 8.0);
        assert_eq!(a.squared_distance(b), 50.0);
    }

    // Direction operations
    #[test]
    fn test_vec3_direction_to() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        assert_eq!(a.direction_to(b), Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_vec3_direction_to_normalized() {
        let a = Vec3::new(1.0, 1.0, 0.0);
        let b = Vec3::new(2.0, 2.0, 0.0);
        let dir = a.direction_to(b);
        assert!(dir.is_normalized());
    }

    // Projection and rejection
    #[test]
    fn test_vec3_project() {
        let a = Vec3::new(1.0, 1.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        assert_eq!(a.project(b), Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_vec3_reject() {
        let a = Vec3::new(1.0, 1.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        assert_eq!(a.reject(b), Vec3::new(0.0, 1.0, 0.0));
    }

    // Reflection
    #[test]
    fn test_vec3_reflect() {
        let v = Vec3::new(1.0, -1.0, 0.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(v.reflect(normal), Vec3::new(1.0, 1.0, 0.0));
    }

    // Interpolation
    #[test]
    fn test_vec3_lerp() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(Vec3::lerp(a, b, 0.5), Vec3::new(2.5, 3.5, 4.5));
    }

    #[test]
    fn test_vec3_slerp() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let result = Vec3::slerp(a, b, 0.5);
        let expected = Vec3::new(0.7071068, 0.7071068, 0.0);
        assert!(result.approx_eq(expected));
    }

    #[test]
    fn test_vec3_slerp_angle() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let result = Vec3::slerp_angle(a, b, 0.5);
        let expected = Vec3::new(0.7071068, 0.7071068, 0.0);
        assert!(result.approx_eq(expected));
    }

    // Barycentric coordinates
    #[test]
    fn test_vec3_barycentric() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        let p = Vec3::new(0.5, 0.5, 0.0);
        let (u, v, w) = Vec3::barycentric(p, a, b, c);
        assert!(epsilon_eq(u, 0.0, 1e-6));
        assert!(epsilon_eq(v, 0.5, 1e-6));
        assert!(epsilon_eq(w, 0.5, 1e-6));
    }

    #[test]
    fn test_vec3_in_triangle() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        let p = Vec3::new(0.25, 0.25, 0.0);
        assert!(Vec3::in_triangle(p, a, b, c));
    }

    // Operator overloading
    #[test]
    fn test_vec3_add() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a + b, Vec3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_vec3_sub() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a - b, Vec3::new(-3.0, -3.0, -3.0));
    }

    #[test]
    fn test_vec3_mul() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a * b, Vec3::new(4.0, 10.0, 18.0));
    }

    #[test]
    fn test_vec3_scalar_mul() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(a * 2.0, Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(2.0 * a, Vec3::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_vec3_div() {
        let a = Vec3::new(4.0, 9.0, 16.0);
        let b = Vec3::new(2.0, 3.0, 4.0);
        assert_eq!(a / b, Vec3::new(2.0, 3.0, 4.0));
    }

    #[test]
    fn test_vec3_scalar_div() {
        let a = Vec3::new(2.0, 4.0, 6.0);
        assert_eq!(a / 2.0, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_vec3_neg() {
        let a = Vec3::new(1.0, -2.0, 3.0);
        assert_eq!(-a, Vec3::new(-1.0, 2.0, -3.0));
    }

    // Edge cases
    #[test]
    fn test_vec3_move_towards_exact() {
        let current = Vec3::new(1.0, 2.0, 3.0);
        let target = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(Vec3::move_towards(current, target, 5.196152), target);
    }

    #[test]
    fn test_vec3_move_towards_partial() {
        let current = Vec3::new(0.0, 0.0, 0.0);
        let target = Vec3::new(3.0, 4.0, 0.0);
        let result = Vec3::move_towards(current, target, 2.5);
        assert_eq!(result, Vec3::new(1.5, 2.0, 0.0));
    }

    #[test]
    fn test_vec3_approx_eq() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(1.000001, 2.000001, 3.000001);
        assert!(a.approx_eq(b));
    }
}