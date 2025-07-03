use crate::mat2::Mat2;
use crate::vec2::Vec2;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f32::consts::PI; // Add `approx = "0.5"` to Cargo.toml for floating-point comparisons

    // Test constants
    const EPSILON: f32 = 1e-6;
    const IDENTITY: Mat2 = Mat2 {
        x: Vec2::new(1.0, 0.0),
        y: Vec2::new(0.0, 1.0),
    };
    const ZERO: Mat2 = Mat2 {
        x: Vec2::zero(),
        y: Vec2::zero(),
    };

    // Helper functions
    fn random_mat2() -> Mat2 {
        Mat2::new(
            Vec2::new(rand::random(), rand::random()),
            Vec2::new(rand::random(), rand::random()),
        )
    }

    // Construction tests
    #[test]
    fn test_construction() {
        // Test basic construction
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        assert_eq!(m.x, Vec2::new(1.0, 2.0));
        assert_eq!(m.y, Vec2::new(3.0, 4.0));

        // Test identity matrix
        assert_eq!(Mat2::identity(), IDENTITY);

        // Test zero matrix
        assert_eq!(Mat2::zero(), ZERO);

        // Test from_diagonal
        let diag = Mat2::from_diagonal(Vec2::new(5.0, 6.0));
        assert_eq!(diag.x, Vec2::new(5.0, 0.0));
        assert_eq!(diag.y, Vec2::new(0.0, 6.0));
    }

    // Rotation matrix tests
    #[test]
    fn test_rotation() {
        // Test 0 rotation
        let rot0 = Mat2::from_rotation(0.0);
        assert_relative_eq!(rot0, IDENTITY, epsilon = EPSILON);

        // Test PI/2 rotation
        let rot90 = Mat2::from_rotation(PI / 2.0);
        let expected90 = Mat2::new(Vec2::new(0.0, 1.0), Vec2::new(-1.0, 0.0));
        assert_relative_eq!(rot90, expected90, epsilon = EPSILON);

        // Test PI rotation
        let rot180 = Mat2::from_rotation(PI);
        let expected180 = Mat2::new(Vec2::new(-1.0, 0.0), Vec2::new(0.0, -1.0));
        assert_relative_eq!(rot180, expected180, epsilon = EPSILON);

        // Test rotation composition (should be additive)
        let angle1 = PI / 4.0;
        let angle2 = PI / 3.0;
        let rot1 = Mat2::from_rotation(angle1);
        let rot2 = Mat2::from_rotation(angle2);
        let rot_composed = Mat2::from_rotation(angle1 + angle2);
        assert_relative_eq!(rot1 * rot2, rot_composed, epsilon = EPSILON);
    }

    // Scaling matrix tests
    #[test]
    fn test_scaling() {
        let scale = Vec2::new(2.0, 3.0);
        let scale_mat = Mat2::from_scale(scale);

        // Test scaling vector
        let v = Vec2::new(4.0, 5.0);
        assert_eq!(scale_mat * v, Vec2::new(8.0, 15.0));

        // Test scaling composition (should be multiplicative)
        let scale2 = Vec2::new(0.5, 0.5);
        let scale_mat2 = Mat2::from_scale(scale2);
        assert_relative_eq!(
            scale_mat * scale_mat2,
            Mat2::from_scale(Vec2::new(1.0, 1.5)),
            epsilon = EPSILON
        );
    }

    // Transpose tests
    #[test]
    fn test_transpose() {
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let transposed = m.transpose();
        assert_eq!(transposed.x, Vec2::new(1.0, 3.0));
        assert_eq!(transposed.y, Vec2::new(2.0, 4.0));

        // Transpose of transpose should be original
        assert_eq!(transposed.transpose(), m);

        // Transpose of identity is identity
        assert_eq!(IDENTITY.transpose(), IDENTITY);
    }

    // Determinant tests
    #[test]
    fn test_determinant() {
        // Test identity matrix
        assert_relative_eq!(IDENTITY.determinant(), 1.0, epsilon = EPSILON);

        // Test zero matrix
        assert_relative_eq!(ZERO.determinant(), 0.0, epsilon = EPSILON);

        // Test specific matrix
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        assert_relative_eq!(m.determinant(), -2.0, epsilon = EPSILON);

        // Test scaling matrix
        let scale = Mat2::from_scale(Vec2::new(2.0, 3.0));
        assert_relative_eq!(scale.determinant(), 6.0, epsilon = EPSILON);

        // Test rotation matrix (should always have det = 1)
        let rot = Mat2::from_rotation(PI / 3.0);
        assert_relative_eq!(rot.determinant(), 1.0, epsilon = EPSILON);
    }

    // Inverse tests
    #[test]
    fn test_inverse() {
        // Test identity matrix
        assert_eq!(IDENTITY.inverse(), Some(IDENTITY));

        // Test zero matrix (should fail)
        assert_eq!(ZERO.inverse(), None);

        // Test specific matrix
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let inv_m = m.inverse().unwrap();
        let expected_inv = Mat2::new(Vec2::new(-2.0, 1.0), Vec2::new(1.5, -0.5));
        assert_relative_eq!(inv_m, expected_inv, epsilon = EPSILON);

        // Test inverse property: m * inv(m) should be identity
        assert_relative_eq!(m * inv_m, IDENTITY, epsilon = EPSILON);
        assert_relative_eq!(inv_m * m, IDENTITY, epsilon = EPSILON);

        // Test scaling matrix inverse
        let scale = Mat2::from_scale(Vec2::new(2.0, 4.0));
        let inv_scale = scale.inverse().unwrap();
        assert_relative_eq!(
            inv_scale,
            Mat2::from_scale(Vec2::new(0.5, 0.25)),
            epsilon = EPSILON
        );

        // Test rotation matrix inverse (should be transpose)
        let rot = Mat2::from_rotation(PI / 5.0);
        let inv_rot = rot.inverse().unwrap();
        assert_relative_eq!(inv_rot, rot.transpose(), epsilon = EPSILON);
    }

    // Operator tests
    #[test]
    fn test_add() {
        let m1 = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let m2 = Mat2::new(Vec2::new(5.0, 6.0), Vec2::new(7.0, 8.0));
        let result = m1 + m2;
        assert_eq!(result.x, Vec2::new(6.0, 8.0));
        assert_eq!(result.y, Vec2::new(10.0, 12.0));

        // Test commutative property
        assert_eq!(m1 + m2, m2 + m1);

        // Test additive identity
        assert_eq!(m1 + ZERO, m1);
    }

    #[test]
    fn test_sub() {
        let m1 = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let m2 = Mat2::new(Vec2::new(5.0, 6.0), Vec2::new(7.0, 8.0));
        let result = m1 - m2;
        assert_eq!(result.x, Vec2::new(-4.0, -4.0));
        assert_eq!(result.y, Vec2::new(-4.0, -4.0));

        // Test subtracting zero
        assert_eq!(m1 - ZERO, m1);

        // Test subtracting self
        assert_eq!(m1 - m1, ZERO);
    }

    #[test]
    fn test_mat_mat_mul() {
        let m1 = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let m2 = Mat2::new(Vec2::new(5.0, 6.0), Vec2::new(7.0, 8.0));
        let result = m1 * m2;
        let expected = Mat2::new(
            Vec2::new(1.0 * 5.0 + 3.0 * 6.0, 2.0 * 5.0 + 4.0 * 6.0),
            Vec2::new(1.0 * 7.0 + 3.0 * 8.0, 2.0 * 7.0 + 4.0 * 8.0),
        );
        assert_eq!(result, expected);

        // Test multiplicative identity
        assert_eq!(m1 * IDENTITY, m1);
        assert_eq!(IDENTITY * m1, m1);

        // Test multiplication by zero
        assert_eq!(m1 * ZERO, ZERO);
        assert_eq!(ZERO * m1, ZERO);

        // Test associativity
        let m3 = random_mat2();
        assert_relative_eq!((m1 * m2) * m3, m1 * (m2 * m3), epsilon = EPSILON);
    }

    #[test]
    fn test_scalar_mul() {
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let scalar = 2.0;
        let result = m * scalar;
        assert_eq!(result.x, Vec2::new(2.0, 4.0));
        assert_eq!(result.y, Vec2::new(6.0, 8.0));

        // Test commutative property
        assert_eq!(m * scalar, scalar * m);

        // Test multiplicative identity
        assert_eq!(m * 1.0, m);

        // Test multiplication by zero
        assert_eq!(m * 0.0, ZERO);
    }

    #[test]
    fn test_scalar_div() {
        let m = Mat2::new(Vec2::new(2.0, 4.0), Vec2::new(6.0, 8.0));
        let scalar = 2.0;
        let result = m / scalar;
        assert_eq!(result.x, Vec2::new(1.0, 2.0));
        assert_eq!(result.y, Vec2::new(3.0, 4.0));

        // Test division by 1
        assert_eq!(m / 1.0, m);

        // Test division by very small number
        let small = 1e-10;
        let large = 1e10;
        assert_relative_eq!((m * large) / small, m * (large / small), epsilon = EPSILON);
    }

    #[test]
    #[should_panic]
    fn test_scalar_div_by_zero() {
        let m = random_mat2();
        let _ = m / 0.0;
    }

    // Matrix-vector multiplication tests
    #[test]
    fn test_mat_vec_mul() {
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let v = Vec2::new(5.0, 6.0);
        let result = m * v;
        assert_eq!(
            result,
            Vec2::new(1.0 * 5.0 + 3.0 * 6.0, 2.0 * 5.0 + 4.0 * 6.0)
        );

        // Test identity transformation
        assert_eq!(IDENTITY * v, v);

        // Test zero transformation
        assert_eq!(ZERO * v, Vec2::zero());

        // Test scaling transformation
        let scale = Mat2::from_scale(Vec2::new(2.0, 3.0));
        assert_eq!(scale * v, Vec2::new(10.0, 18.0));

        // Test rotation transformation
        let rot = Mat2::from_rotation(PI / 2.0);
        let v = Vec2::new(1.0, 0.0);
        let rotated = rot * v;
        assert_relative_eq!(rotated, Vec2::new(0.0, 1.0), epsilon = EPSILON);
    }

    // Approximate equality tests
    #[test]
    fn test_approx_eq() {
        let m1 = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let m2 = Mat2::new(
            Vec2::new(1.0 + EPSILON / 2.0, 2.0),
            Vec2::new(3.0, 4.0 - EPSILON / 2.0),
        );
        assert!(m1.approx_eq(m2, 1e-6));

        let m3 = Mat2::new(Vec2::new(1.0 + 2.0 * EPSILON, 2.0), Vec2::new(3.0, 4.0));
        assert!(!m1.approx_eq(m3, 1e-6));
    }

    // Edge case tests
    #[test]
    fn test_edge_cases() {
        // Test with very large numbers
        let large = 1e20;
        let m_large = Mat2::new(Vec2::new(large, 0.0), Vec2::new(0.0, large));
        let v_large = Vec2::new(large, large);
        let result = m_large * v_large;
        assert_relative_eq!(result.x, large * large, epsilon = 1e-6);
        assert_relative_eq!(result.y, large * large, epsilon = 1e-6);

        // Test with very small numbers
        let small = 1e-20;
        let m_small = Mat2::new(Vec2::new(small, 0.0), Vec2::new(0.0, small));
        let v_small = Vec2::new(small, small);
        assert_eq!(m_small * v_small, Vec2::new(small * small, small * small));

        // Test with NaN and infinity
        let m_nan = Mat2::new(Vec2::new(f32::NAN, 0.0), Vec2::new(0.0, 1.0));
        let v_inf = Vec2::new(f32::INFINITY, 1.0);
        let result = m_nan * v_inf;

        // First component is NAN * INFINITY + 0 * 1 = NAN
        assert!(result.x.is_nan());

        // Second component is 0 * INFINITY + 1 * 1 = NAN (since 0*INF is NAN)
        assert!(result.y.is_nan());

        // Test case that should work without NaN
        let m_identity = Mat2::identity();
        let result = m_identity * v_inf;
        assert_eq!(result, Vec2::new(f32::INFINITY, 1.0));
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use approx::assert_relative_eq;

        #[test]
        fn test_lu_decomposition() {
            let m = Mat2::new(Vec2::new(2.0, 1.0), Vec2::new(1.0, 2.0));
            let (p, l, u) = m.lu_decompose();

            // Verify PA = LU
            let pa = p * m;
            let lu = l * u;
            assert_relative_eq!(pa.x.x, lu.x.x, epsilon = 1e-6);
            assert_relative_eq!(pa.x.y, lu.x.y, epsilon = 1e-6);
            assert_relative_eq!(pa.y.x, lu.y.x, epsilon = 1e-6);
            assert_relative_eq!(pa.y.y, lu.y.y, epsilon = 1e-6);

            // Verify L is lower triangular with 1s on diagonal
            assert_relative_eq!(l.x.x, 1.0, epsilon = 1e-6);
            assert_relative_eq!(l.y.y, 1.0, epsilon = 1e-6);
            assert_relative_eq!(l.x.y, 0.0, epsilon = 1e-6);

            // Verify U is upper triangular
            assert_relative_eq!(u.y.x, 0.0, epsilon = 1e-6);

            // Verify P is permutation matrix
            assert_relative_eq!(p.determinant().abs(), 1.0, epsilon = 1e-6);
        }

        #[test]
        fn test_qr_decomposition() {
            let m = Mat2::new(Vec2::new(3.0, -1.0), Vec2::new(0.0, 2.0));
            let (q, r) = m.qr_decompose();

            // Verify A = QR
            assert_relative_eq!(q * r, m, epsilon = 1e-6);

            // Verify Q is orthogonal
            assert_relative_eq!(q * q.transpose(), Mat2::identity(), epsilon = 1e-6);
            assert_relative_eq!(q.transpose() * q, Mat2::identity(), epsilon = 1e-6);

            // Verify R is upper triangular
            assert_relative_eq!(r.y.x, 0.0, epsilon = 1e-6);
        }
    }
}
