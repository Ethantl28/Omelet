use crate::mat3::Mat3;
use crate::vec::{Vec2, Vec3, Vec4};

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    // A small constant for floating-point comparisons.
    const EPSILON: f32 = 1e-6;

    // Helper for single float comparisons
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_utility_and_check_methods() {
        let m = Mat3::from_scale(Vec2::new(-2.0, 3.0));

        // Test abs() and signum()
        assert_eq!(m.abs().get_scale(), Vec2::new(2.0, 3.0));
        let s = m.signum();
        dbg!(m);
        dbg!(s);

        // Each component of `signum()` should return -1.0, 0.0, or 1.0 depending on its sign
        assert_eq!(s.col0, Vec3::new(-1.0, 0.0, 0.0));
        assert_eq!(s.col1, Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(s.col2, Vec3::new(0.0, 0.0, 1.0));

        // Test .lerp() and .lerp_between()
        let a = Mat3::identity();
        let b = Mat3::from_translation(Vec2::new(10.0, 10.0));
        let mid1 = a.lerp(b, 0.5);
        let mid2 = Mat3::lerp_between(a, b, 0.5);
        let expected_mid = Mat3::from_translation(Vec2::new(5.0, 5.0));
        assert!(mid1.approx_eq_eps(expected_mid, EPSILON));
        assert!(mid2.approx_eq_eps(expected_mid, EPSILON));

        // Test .get_diagonal() and .trace()
        assert_eq!(m.get_diagonal(), Vec2::new(-2.0, 3.0));
        assert_eq!(m.trace(), -2.0 + 3.0 + 1.0);

        // Test checks: is_invertible, is_identity, is_affine, has_mirroring
        assert!(m.is_invertible());
        assert!(!Mat3::zero().is_invertible());
        assert!(Mat3::identity().is_identity());
        assert!(!m.is_identity());
        assert!(m.is_affine());
        assert!(m.has_mirroring());
        assert!(!Mat3::identity().has_mirroring());
    }

    #[test]
    fn test_graphics_matrices() {
        // Test .ortho()
        let proj = Mat3::ortho(0.0, 800.0, 600.0, 0.0);
        let top_left = proj.transform_point(Vec2::new(0.0, 0.0));
        let bottom_right = proj.transform_point(Vec2::new(800.0, 600.0));
        assert!(top_left.approx_eq_eps(Vec2::new(-1.0, 1.0), EPSILON));
        assert!(bottom_right.approx_eq_eps(Vec2::new(1.0, -1.0), EPSILON));

        // Test .look_at()
        let eye = Vec2::new(100.0, 50.0);
        let target = Vec2::new(100.0, 60.0); // Look along +Y axis
        let view = Mat3::look_at(eye, target);

        // A point at the "eye" should be transformed to the origin
        assert!(
            view.transform_point(eye)
                .approx_eq_eps(Vec2::zero(), EPSILON)
        );

        // A point 1 unit "in front" of the eye (along the target direction)
        // should be transformed to (0, 1) in view space (along the +Y fwd axis)
        let point_in_front = eye + (target - eye).normalize();
        let transformed_point = view.transform_point(point_in_front);
        assert!(transformed_point.approx_eq_eps(Vec2::new(0.0, 1.0), EPSILON));
    }

    #[test]
    fn test_transformation_constructors() {
        // Test ::from_translation()
        let t = Mat3::from_translation(Vec2::new(10.0, -20.0));
        assert_eq!(t.col2, Vec3::new(10.0, -20.0, 1.0));
        assert_eq!(t.col0, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(t.col1, Vec3::new(0.0, 1.0, 0.0));

        // Test ::from_scale()
        let s = Mat3::from_scale(Vec2::new(2.0, 3.0));
        assert_eq!(s.col0, Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(s.col1, Vec3::new(0.0, 3.0, 0.0));
        assert_eq!(s.col2, Vec3::new(0.0, 0.0, 1.0));

        // Test ::from_angle_z()
        let r = Mat3::from_angle_z(PI / 2.0); // 90 degrees
        assert!(r.col0.approx_eq_eps(Vec3::new(0.0, 1.0, 0.0), EPSILON));
        assert!(r.col1.approx_eq_eps(Vec3::new(-1.0, 0.0, 0.0), EPSILON));
        assert_eq!(r.col2, Vec3::new(0.0, 0.0, 1.0));

        // Test ::from_shear()
        let shear = Mat3::from_shear(Vec2::new(2.0, 0.0));
        let p = Vec2::new(1.0, 1.0);
        let transformed = shear.transform_point(p);
        assert_eq!(transformed, Vec2::new(3.0, 1.0)); // x' = x + shear_x * y

        // Test ::from_angle_axis()
        let rot_x_90 = Mat3::from_angle_axis(PI / 2.0, Vec3::new(1.0, 0.0, 0.0));
        let v = Vec3::new(0.0, 1.0, 0.0);
        let rotated_v = rot_x_90 * v;
        assert!(rotated_v.approx_eq_eps(Vec3::new(0.0, 0.0, 1.0), EPSILON));
    }

    #[test]
    fn test_core_matrix_ops() {
        let m = Mat3::from_rows(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.0, 1.0, 4.0),
            Vec3::new(5.0, 6.0, 0.0),
        );

        // Test .transpose()
        let mt = m.transpose();
        assert_eq!(mt.col0, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(m.transpose().transpose(), m);

        // Test .determinant()
        assert!(approx(m.determinant(), 1.0));

        // Test .inverse() and .adjugate()
        let m_inv = m.inverse().unwrap();
        let identity = m * m_inv;
        assert!(identity.approx_eq_eps(Mat3::identity(), EPSILON));
        assert!(m.adjugate().approx_eq_eps(m_inv * m.determinant(), EPSILON));

        // Test .inverse() on a singular matrix
        let singular = Mat3::new(
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(2.0, 2.0, 2.0),
            Vec3::new(3.0, 3.0, 3.0),
        );
        assert!(singular.inverse().is_none());

        // Test .inverse_or_identity()
        assert_eq!(singular.inverse_or_identity(), Mat3::identity());
        assert_eq!(m.inverse_or_identity(), m_inv);
    }

    #[test]
    fn test_accessors_and_indexing() {
        let mut m = Mat3::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
        );

        // Test .row()
        assert_eq!(m.row(0), Vec3::new(1.0, 4.0, 7.0));
        assert_eq!(m.row(1), Vec3::new(2.0, 5.0, 8.0));
        assert_eq!(m.row(2), Vec3::new(3.0, 6.0, 9.0));

        // Test Index read
        assert_eq!(m[0], Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(m[1], Vec3::new(4.0, 5.0, 6.0));

        // Test IndexMut write
        m[2] = Vec3::new(10.0, 11.0, 12.0);
        assert_eq!(m.col2, Vec3::new(10.0, 11.0, 12.0));
    }

    #[test]
    fn test_transformations_and_decomposition() {
        let t = Vec2::new(10.0, -20.0);
        let r = PI / 6.0; // 30 degrees
        let s = Vec2::new(2.0, 0.5);
        let m = Mat3::from_translation(t) * Mat3::from_angle_z(r) * Mat3::from_scale(s);

        // Test decomposition
        assert!(m.get_translation().approx_eq_eps(t, EPSILON));
        assert!(m.get_scale().approx_eq_eps(s, EPSILON));
        assert!(approx(m.get_rotation(), r));
        let (dec_t, dec_r, dec_s) = m.decompose();
        assert!(dec_t.approx_eq_eps(t, EPSILON));
        assert!(dec_s.approx_eq_eps(s, EPSILON));
        assert!(approx(dec_r, r));

        // Test .transform_point() vs .transform_vector()
        let point = Vec2::new(1.0, 2.0);
        let vector = Vec2::new(1.0, 2.0);
        let transformed_point = m.transform_point(point);
        let transformed_vector = m.transform_vector(vector);

        // Check that translation was applied to point but not vector
        assert!((transformed_point - transformed_vector).approx_eq_eps(t, EPSILON));

        // Test .transform_aabb()
        let rot_45 = Mat3::from_angle_z(PI / 4.0);
        let min = Vec2::new(-1.0, -1.0);
        let max = Vec2::new(1.0, 1.0);
        let (new_min, new_max) = rot_45.transform_aabb(min, max);
        let sqrt2 = 2.0_f32.sqrt();
        assert!(new_min.approx_eq_eps(Vec2::new(-sqrt2, -sqrt2), EPSILON));
        assert!(new_max.approx_eq_eps(Vec2::new(sqrt2, sqrt2), EPSILON));
    }

    #[test]
    fn test_builder_methods() {
        let m_start = Mat3::from_scale(Vec2::new(2.0, 2.0));
        let t = Vec2::new(10.0, 0.0);
        let r = PI;

        // Test .with_... (consuming)
        let m_with = m_start.with_translation(t).with_rotation(r);
        let m_expected = Mat3::from_angle_z(r) * Mat3::from_translation(t) * m_start;
        assert!(m_with.approx_eq_eps(m_expected, EPSILON));

        // Test .apply_... (mutating)
        let mut m_apply = m_start;
        m_apply.apply_translation(t);
        m_apply.apply_rotation(r);
        assert!(m_apply.approx_eq_eps(m_expected, EPSILON));
    }

    #[test]
    fn test_conversions_and_operators() {
        let m3 = Mat3::from_trs(Vec2::new(10.0, 20.0), PI, Vec2::new(2.0, 2.0), Vec2::zero());
        let m4 = m3.to_mat4_affine();

        // Corrected test for to_mat4_affine
        let p2 = Vec2::new(5.0, 5.0);
        let transformed_p2 = m3.transform_point(p2);

        // Transform a corresponding Vec4 with the Mat4
        let p4 = Vec4::new(p2.x, p2.y, 0.0, 1.0);
        let transformed_p4 = m4 * p4;

        // The XY of the transformed Vec4 should match the transformed Vec2
        assert!(
            transformed_p2.approx_eq_eps(Vec2::new(transformed_p4.x, transformed_p4.y), EPSILON)
        );

        let m = Mat3::from_angle_z(1.23);
        assert_eq!(Mat3::from_2d_array(m.to_array_2d_row_major()), m);
        assert_eq!(Mat3::from_tuple(m.to_tuple_row_major()), m);
    }

    #[test]
    #[should_panic]
    fn test_index_panic() {
        let m = Mat3::identity();
        let _ = m[3];
    }

    #[test]
    #[should_panic]
    fn test_index_mut_panic() {
        let mut m = Mat3::identity();
        m[3] = Vec3::zero();
    }
}
