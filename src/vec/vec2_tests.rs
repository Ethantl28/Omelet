
use super::vec2::Vec2;

///Vec2 tests
#[cfg(test)]
mod tests {
    use super::*;

    // Basic construction and properties
    #[test]
    fn test_vec2_construction() {
        let v = Vec2::new(1.0, 2.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
    }

    #[test]
    fn test_vec2_default() {
        let v = Vec2::new(0.0, 0.0);
        assert_eq!(v.length(), 0.0);
        assert_eq!(v.squared_length(), 0.0);
    }

    // Vector operations
    #[test]
    fn test_vec2_length() {
        let v = Vec2::new(3.0, 4.0);
        assert_eq!(v.length(), 5.0);
    }

    #[test]
    fn test_vec2_squared_length() {
        let v = Vec2::new(3.0, 4.0);
        assert_eq!(v.squared_length(), 25.0);
    }

    #[test]
    fn test_vec2_normalize() {
        let v = Vec2::new(3.0, 4.0).normalize();
        assert!(v.is_normalized());
        assert!(v.is_normalized_fast());
        assert!(v.approx_eq(Vec2::new(0.6, 0.8), 1e-6));
    }

    #[test]
    fn test_vec2_normalize_zero() {
        let v = Vec2::new(0.0, 0.0).normalize();
        assert_eq!(v, Vec2::new(0.0, 0.0));
    }

    // Dot product
    #[test]
    fn test_vec2_dot() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert_eq!(a.dot(b), 11.0);
    }

    // Distance operations
    #[test]
    fn test_vec2_distance() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(4.0, 6.0);
        assert_eq!(a.distance(b), 5.0);
    }

    #[test]
    fn test_vec2_squared_distance() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(4.0, 6.0);
        assert_eq!(a.squared_distance(b), 25.0);
    }

    // Direction operations
    #[test]
    fn test_vec2_direction_to() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(2.0, 0.0);
        assert_eq!(a.direction_to(b), Vec2::new(1.0, 0.0));
    }

    #[test]
    fn test_vec2_direction_to_normalized() {
        let a = Vec2::new(1.0, 1.0);
        let b = Vec2::new(2.0, 2.0);
        let dir = a.direction_to(b);
        assert!(dir.is_normalized());
    }

    // Projection and rejection
    #[test]
    fn test_vec2_project() {
        let a = Vec2::new(1.0, 1.0);
        let b = Vec2::new(1.0, 0.0);
        assert_eq!(a.project(b), Vec2::new(1.0, 0.0));
    }

    #[test]
    fn test_vec2_reject() {
        let a = Vec2::new(1.0, 1.0);
        let b = Vec2::new(1.0, 0.0);
        assert_eq!(a.reject(b), Vec2::new(0.0, 1.0));
    }

    // Reflection
    #[test]
    fn test_vec2_reflect() {
        let v = Vec2::new(1.0, -1.0);
        let normal = Vec2::new(0.0, 1.0);
        assert_eq!(v.reflect(normal), Vec2::new(1.0, 1.0));
    }

    // Interpolation
    #[test]
    fn test_vec2_lerp() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert_eq!(Vec2::lerp(a, b, 0.5), Vec2::new(2.0, 3.0));
    }

    #[test]
    fn test_vec2_slerp() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        let result = Vec2::slerp(a, b, 0.5);
        let expected = Vec2::new(0.7071068, 0.7071068);
        assert!(result.approx_eq(expected, 1e-6));
    }

    #[test]
    fn test_vec2_slerp_angle() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        let result = Vec2::slerp_angle(a, b, 0.5);
        let expected = Vec2::new(0.7071068, 0.7071068);
        assert!(result.approx_eq(expected, 1e-6));
    }

    // Operator overloading
    #[test]
    fn test_vec2_add() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert_eq!(a + b, Vec2::new(4.0, 6.0));
    }

    #[test]
    fn test_vec2_sub() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert_eq!(a - b, Vec2::new(-2.0, -2.0));
    }

    #[test]
    fn test_vec2_mul() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert_eq!(a * b, Vec2::new(3.0, 8.0));
    }

    #[test]
    fn test_vec2_scalar_mul() {
        let a = Vec2::new(1.0, 2.0);
        assert_eq!(a * 2.0, Vec2::new(2.0, 4.0));
        assert_eq!(2.0 * a, Vec2::new(2.0, 4.0));
    }

    #[test]
    fn test_vec2_div() {
        let a = Vec2::new(4.0, 9.0);
        let b = Vec2::new(2.0, 3.0);
        assert_eq!(a / b, Vec2::new(2.0, 3.0));
    }

    #[test]
    fn test_vec2_scalar_div() {
        let a = Vec2::new(2.0, 4.0);
        assert_eq!(a / 2.0, Vec2::new(1.0, 2.0));
    }

    #[test]
    fn test_vec2_neg() {
        let a = Vec2::new(1.0, -2.0);
        assert_eq!(-a, Vec2::new(-1.0, 2.0));
    }

    // Edge cases
    #[test]
    fn test_vec2_move_towards_exact() {
        let current = Vec2::new(1.0, 2.0);
        let target = Vec2::new(4.0, 6.0);
        assert_eq!(Vec2::move_towards(current, target, 5.0), target);
    }

    #[test]
    fn test_vec2_move_towards_partial() {
        let current = Vec2::new(0.0, 0.0);
        let target = Vec2::new(3.0, 4.0);
        let result = Vec2::move_towards(current, target, 2.5);
        assert_eq!(result, Vec2::new(1.5, 2.0));
    }

    #[test]
    fn test_vec2_approx_eq() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(1.000001, 2.000001);
        assert!(a.approx_eq(b, 1e-6));
    }
}