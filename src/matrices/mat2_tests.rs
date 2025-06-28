use crate::mat2::Mat2;
use crate::vec2::Vec2;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

    // Construction Tests
    #[test]
    fn test_new() {
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        assert_eq!(m.x, Vec2::new(1.0, 2.0));
        assert_eq!(m.y, Vec2::new(3.0, 4.0));
    }

    #[test]
    fn test_identity() {
        let ident = Mat2::identity();
        assert_eq!(ident.x, Vec2::new(1.0, 0.0));
        assert_eq!(ident.y, Vec2::new(0.0, 1.0));
    }

    #[test]
    fn test_zero() {
        let zero = Mat2::zero();
        assert_eq!(zero.x, Vec2::zero());
        assert_eq!(zero.y, Vec2::zero());
    }

    // Factory Method Tests
    #[test]
    fn test_from_diagonal() {
        let diag = Mat2::from_diagonal(Vec2::new(2.0, 3.0));
        assert_eq!(diag.x, Vec2::new(2.0, 0.0));
        assert_eq!(diag.y, Vec2::new(0.0, 3.0));
    }

    #[test]
    fn test_from_rotation() {
        let rot = Mat2::from_rotation(FRAC_PI_2); // 90 degrees
        assert!(rot.x.approx_eq(Vec2::new(0.0, 1.0)));
        assert!(rot.y.approx_eq(Vec2::new(-1.0, 0.0)));
    }

    #[test]
    fn test_from_scale() {
        let scale = Mat2::from_scale(Vec2::new(2.0, 3.0));
        assert_eq!(scale.x, Vec2::new(2.0, 0.0));
        assert_eq!(scale.y, Vec2::new(0.0, 3.0));
    }

    // Matrix Operations
    #[test]
    fn test_transpose() {
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let transposed = m.transpose();
        assert_eq!(transposed.x, Vec2::new(1.0, 3.0));
        assert_eq!(transposed.y, Vec2::new(2.0, 4.0));
    }

    #[test]
    fn test_determinant() {
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        assert_eq!(m.determinant(), -2.0);
    }

    #[test]
    fn test_inverse() {
        let m = Mat2::new(Vec2::new(4.0, 3.0), Vec2::new(3.0, 2.0));
        let inv = m.inverse().unwrap();
        
        // Verify m * inv = identity
        let identity = m * inv;
        assert!(identity.approx_eq(Mat2::identity()));
    }

    #[test]
    fn test_inverse_singular() {
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(2.0, 4.0)); // det = 0
        assert!(m.inverse().is_none());
    }

    // Vector Multiplication
    #[test]
    fn test_mul_vec2() {
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let v = Vec2::new(5.0, 6.0);
        let result = m.mul_vec2(v);
        assert_eq!(result, Vec2::new(23.0, 34.0)); // 1*5 + 3*6, 2*5 + 4*6
    }

    // Matrix Multiplication
    #[test]
    fn test_matrix_multiplication() {
        let a = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let b = Mat2::new(Vec2::new(5.0, 6.0), Vec2::new(7.0, 8.0));
        let result = a * b;
        
        // Manual calculation:
        // Column 1: [1*5 + 3*6, 2*5 + 4*6] = [23, 34]
        // Column 2: [1*7 + 3*8, 2*7 + 4*8] = [31, 46]
        assert_eq!(result.x, Vec2::new(23.0, 34.0));
        assert_eq!(result.y, Vec2::new(31.0, 46.0));
    }

    #[test]
    fn test_multiplication_identity() {
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        assert_eq!(m * Mat2::identity(), m);
        assert_eq!(Mat2::identity() * m, m);
    }

    // Scalar Operations
    #[test]
    fn test_scalar_multiplication() {
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let result = m * 2.0;
        assert_eq!(result.x, Vec2::new(2.0, 4.0));
        assert_eq!(result.y, Vec2::new(6.0, 8.0));
    }

    #[test]
    fn test_scalar_division() {
        let m = Mat2::new(Vec2::new(2.0, 4.0), Vec2::new(6.0, 8.0));
        let result = m / 2.0;
        assert_eq!(result.x, Vec2::new(1.0, 2.0));
        assert_eq!(result.y, Vec2::new(3.0, 4.0));
    }

    // Approximate Equality
    #[test]
    fn test_approx_eq() {
        let a = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let b = Mat2::new(Vec2::new(1.000001, 2.000001), Vec2::new(3.000001, 4.000001));
        assert!(a.approx_eq(b));
    }

    // Rotation Verification
    #[test]
    fn test_rotation_90_degrees() {
        let rot = Mat2::from_rotation(FRAC_PI_2);
        let v = Vec2::new(1.0, 0.0);
        let rotated = rot.mul_vec2(v);
        assert!(rotated.approx_eq(Vec2::new(0.0, 1.0)));
    }

    #[test]
    fn test_rotation_composition() {
        let rot1 = Mat2::from_rotation(FRAC_PI_4); // 45°
        let rot2 = Mat2::from_rotation(FRAC_PI_4); // 45°
        let combined = rot1 * rot2;
        let expected = Mat2::from_rotation(FRAC_PI_2); // 90°
        assert!(combined.approx_eq(expected));
    }
}