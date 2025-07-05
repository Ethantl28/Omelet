use crate::mat2::Mat2;
use crate::vec2::Vec2;

#[cfg(test)]
mod tests {
    use approx::{AbsDiffEq, RelativeEq};

    use super::*;
    use core::f32::consts::PI;

    // Helper for approx equality with a small epsilon
    fn assert_approx_eq(a: f32, b: f32, eps: f32) {
        assert!((a - b).abs() < eps, "{} not approx equal to {}", a, b);
    }

    #[test]
    fn test_new_and_identity_and_zero() {
        let v1 = Vec2::new(1.0, 2.0);
        let v2 = Vec2::new(3.0, 4.0);
        let m = Mat2::new(v1, v2);
        assert_eq!(m.x, v1);
        assert_eq!(m.y, v2);

        let id = Mat2::identity();
        assert_eq!(id.x, Vec2::new(1.0, 0.0));
        assert_eq!(id.y, Vec2::new(0.0, 1.0));

        let zero = Mat2::zero();
        assert_eq!(zero.x, Vec2::zero());
        assert_eq!(zero.y, Vec2::zero());
    }

    #[test]
    fn test_from_diagonal_and_from_scale() {
        let diag = Vec2::new(5.0, 10.0);
        let m = Mat2::from_diagonal(diag);
        assert_eq!(m.x, Vec2::new(5.0, 0.0));
        assert_eq!(m.y, Vec2::new(0.0, 10.0));

        let scale = Vec2::new(2.0, 3.0);
        let scaled = Mat2::from_scale(scale);
        assert_eq!(scaled, Mat2::from_diagonal(scale));
    }

    #[test]
    fn test_from_rotation() {
        let m = Mat2::from_rotation(PI / 2.0);
        let expected = Mat2::new(Vec2::new(0.0, 1.0), Vec2::new(-1.0, 0.0));
        assert!(m.approx_eq(expected));
    }

    #[test]
    fn test_transpose() {
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let t = m.transpose();
        assert_eq!(t, Mat2::new(Vec2::new(1.0, 3.0), Vec2::new(2.0, 4.0)));
    }

    #[test]
    fn test_determinant() {
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        assert_eq!(m.determinant(), 1.0 * 4.0 - 2.0 * 3.0);
    }

    #[test]
    fn test_inverse_some_and_none() {
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let inv = m.inverse().unwrap();
        let det = m.determinant();
        let expected = Mat2::new(
            Vec2::new(4.0, -2.0) * (1.0 / det),
            Vec2::new(-3.0, 1.0) * (1.0 / det),
        );
        assert!(inv.approx_eq(expected));

        // Singular matrix (det=0)
        let singular = Mat2::zero();
        assert!(singular.inverse().is_none());
    }

    #[test]
    fn test_approx_eq_and_approx_eq_eps() {
        let m1 = Mat2::identity();
        let m2 = Mat2::identity() * 1.000001;
        assert!(m1.approx_eq(m2));
        assert!(m1.approx_eq_eps(m2, 1e-5));
        assert!(!m1.approx_eq_eps(m2, 1e-7));
    }

    #[test]
    fn test_is_finite_and_is_nan() {
        let m = Mat2::identity();
        assert!(m.is_finite());
        assert!(!m.is_nan());

        let nan_mat = Mat2::new(Vec2::new(f32::NAN, 0.0), Vec2::zero());
        assert!(!nan_mat.is_finite());
        assert!(nan_mat.is_nan());
    }

    #[test]
    fn test_adjugate() {
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let adj = m.adjugate();
        let expected = Mat2::new(Vec2::new(4.0, -2.0), Vec2::new(-3.0, 1.0));
        assert_eq!(adj, expected);
    }

    #[test]
    fn test_trace_and_diagonal_scale() {
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        assert_eq!(m.trace(), 5.0);
        assert_eq!(m.diagonal_scale(), Vec2::new(1.0, 4.0));
    }

    #[test]
    fn test_get_scale_and_get_rotation() {
        let scale = Vec2::new(3.0, 4.0);
        let m = Mat2::from_scale(scale);
        let s = m.get_scale();
        assert_eq!(s, scale);

        // Rotation 90 degrees
        let rot = Mat2::from_rotation(PI / 2.0);
        let angle = rot.get_rotation();
        assert_approx_eq(angle, PI / 2.0, 1e-6);

        // Zero scale returns zero rotation
        let zero = Mat2::zero();
        assert_eq!(zero.get_rotation(), 0.0);
    }

    #[test]
    fn test_decompose() {
        let scale = Vec2::new(2.0, 3.0);
        let rot_angle = PI / 3.0;
        let rot = Mat2::from_rotation(rot_angle);
        let m = rot * Mat2::from_scale(scale);
        let (decomp_scale, decomp_rot) = m.decompose();

        assert_approx_eq(decomp_scale.x, scale.x, 1e-5);
        assert_approx_eq(decomp_scale.y, scale.y, 1e-5);
        assert_approx_eq(decomp_rot, rot_angle, 1e-5);
    }

    #[test]
    #[should_panic(expected = "Mat2 column index out of bounds")]
    fn test_col_panic() {
        let m = Mat2::identity();
        let _ = m.col(2); // invalid index should panic
    }

    #[test]
    fn test_col() {
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        assert_eq!(m.col(0), Vec2::new(1.0, 3.0));
        assert_eq!(m.col(1), Vec2::new(2.0, 4.0));
    }

    #[test]
    fn test_from_cols() {
        let x = Vec2::new(5.0, 6.0);
        let y = Vec2::new(7.0, 8.0);
        let m = Mat2::from_cols(x, y);
        assert_eq!(m.x, x);
        assert_eq!(m.y, y);
    }

    #[test]
    fn test_is_orthogonal() {
        let id = Mat2::identity();
        assert!(id.is_orthogonal(1e-6));

        let rot = Mat2::from_rotation(PI / 4.0);
        assert!(rot.is_orthogonal(1e-6));

        let not_ortho = Mat2::new(Vec2::new(1.0, 1.0), Vec2::new(0.0, 1.0));
        assert!(!not_ortho.is_orthogonal(1e-3));
    }

    #[test]
    fn test_triangular_checks() {
        let lower = Mat2::new(Vec2::new(1.0, 0.0), Vec2::new(2.0, 3.0));
        assert!(lower.is_lower_triangular(1e-6));
        assert!(!lower.is_upper_triangular(1e-6));

        let upper = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(0.0, 3.0));
        assert!(upper.is_upper_triangular(1e-6));
        assert!(!upper.is_lower_triangular(1e-6));
    }

    #[test]
    fn test_to_array_variants() {
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        assert_eq!(m.to_array_2d_row_major(), [[1.0, 3.0], [2.0, 4.0]]);
        assert_eq!(m.to_array_row_major(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m.to_array_2d_col_major(), [[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(m.to_array_col_major(), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_operator_add_sub_mul_div() {
        let a = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let b = Mat2::new(Vec2::new(4.0, 3.0), Vec2::new(2.0, 1.0));
        let c = a + b;
        assert_eq!(c, Mat2::new(Vec2::new(5.0, 5.0), Vec2::new(5.0, 5.0)));

        let d = a - b;
        assert_eq!(d, Mat2::new(Vec2::new(-3.0, -1.0), Vec2::new(1.0, 3.0)));

        let e = a * b;
        // Manual multiply check
        let expected = Mat2::new(
            Vec2::new(1.0 * 4.0 + 3.0 * 3.0, 2.0 * 4.0 + 4.0 * 3.0),
            Vec2::new(1.0 * 2.0 + 3.0 * 1.0, 2.0 * 2.0 + 4.0 * 1.0),
        );
        assert_eq!(e, expected);

        let f = a * 2.0;
        assert_eq!(f, Mat2::new(Vec2::new(2.0, 4.0), Vec2::new(6.0, 8.0)));

        let g = 2.0 * a;
        assert_eq!(g, f);

        let h = f / 2.0;
        assert_eq!(h, a);
    }

    #[test]
    #[should_panic(expected = "Division by 0")]
    fn test_div_by_zero_panic() {
        let m = Mat2::identity();
        let _ = m / 0.0;
    }

    #[test]
    fn test_mul_vec2() {
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let v = Vec2::new(1.0, 1.0);
        let result = m * v;
        // (1*1 + 3*1, 2*1 + 4*1) = (4,6)
        assert_eq!(result, Vec2::new(4.0, 6.0));
    }

    #[test]
    fn test_approx_traits() {
        let a = Mat2::identity();
        let b = Mat2::identity() * 1.0000001;
        assert!(a.abs_diff_eq(&b, 1e-5));
        assert!(a.relative_eq(&b, 1e-5, 1e-5));
    }

    #[test]
    fn test_default() {
        let default = Mat2::default();
        assert_eq!(default, Mat2::zero());
    }

    #[test]
    #[should_panic(expected = "Mat2 row index out of bounds")]
    fn test_index_panic() {
        let m = Mat2::identity();
        let _ = m[2];
    }

    #[test]
    fn test_index() {
        let m = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        assert_eq!(m[0], Vec2::new(1.0, 2.0));
        assert_eq!(m[1], Vec2::new(3.0, 4.0));
    }

    #[test]
    #[should_panic(expected = "Mat2 row index out of bounds")]
    fn test_index_mut_panic() {
        let mut m = Mat2::identity();
        let _ = &mut m[2];
    }

    #[test]
    fn test_index_mut() {
        let mut m = Mat2::identity();
        m[0] = Vec2::new(5.0, 6.0);
        assert_eq!(m[0], Vec2::new(5.0, 6.0));
    }

    #[test]
    fn test_display_format() {
        let m = Mat2::new(Vec2::new(1.23456, 2.34567), Vec2::new(3.45678, 4.56789));
        let s = format!("{}", m);
        assert!(s.contains("[[1.2346, 3.4568],"));
        assert!(s.contains("[2.3457, 4.5679]]"));
    }

    #[test]
    fn test_to_mat3() {
        let m2 = Mat2::new(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let m3 = m2.to_mat3();

        // Check first column
        assert_eq!(m3.x.x, 1.0);
        assert_eq!(m3.x.y, 2.0);
        assert_eq!(m3.x.z, 0.0);

        // Check second column
        assert_eq!(m3.y.x, 3.0);
        assert_eq!(m3.y.y, 4.0);
        assert_eq!(m3.y.z, 0.0);

        // Check third column (identity bottom row)
        assert_eq!(m3.z.x, 0.0);
        assert_eq!(m3.z.y, 0.0);
        assert_eq!(m3.z.z, 1.0);
    }
}
