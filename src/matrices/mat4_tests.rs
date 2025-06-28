use crate::mat4::Mat4;
use crate::vec::{vec3::Vec3, vec4::Vec4};
use crate::utils::epsilon_eq;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

    // ... existing tests ...

    #[test]
    fn test_inverse_identity() {
        let m = Mat4::identity();
        let inv = m.inverse().expect("Identity matrix should be invertible");
        assert!(inv.approx_eq(m), "Inverse of identity should be identity");
    }

    #[test]
    fn test_inverse_translation() {
        let trans = Mat4::from_translation(Vec3::new(5.0, -3.0, 2.0));
        let inv = trans.inverse().expect("Translation matrix should be invertible");
        
        // Verify inverse translation works
        let point = Vec3::new(1.0, 2.0, 3.0);
        let translated = trans.transform_point(point);
        let restored = inv.transform_point(translated);
        assert!(restored.approx_eq(point), "Inverse should undo translation");
    }

    #[test]
    fn test_inverse_rotation() {
        let rot = Mat4::from_rotation_x(FRAC_PI_2);
        let inv = rot.inverse().expect("Rotation matrix should be invertible");
        
        // Verify inverse rotation works
        let v = Vec3::new(0.0, 1.0, 0.0);
        let rotated = rot.transform_vector(v);
        let restored = inv.transform_vector(rotated);
        assert!(restored.approx_eq(v), "Inverse should undo rotation");
        
        // Inverse of rotation should equal its transpose
        assert!(inv.approx_eq(rot.transpose()), "Inverse of rotation should equal transpose");
    }

    #[test]
    fn test_inverse_scale() {
        let scale = Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0));
        let inv = scale.inverse().expect("Scale matrix should be invertible");
        
        // Verify inverse scaling works
        let v = Vec3::new(1.0, 2.0, 3.0);
        let scaled = scale.transform_vector(v);
        let restored = inv.transform_vector(scaled);
        assert!(restored.approx_eq(v), "Inverse should undo scaling");
        
        // Check inverse has reciprocal scales
        let expected = Mat4::from_scale(Vec3::new(0.5, 1.0/3.0, 0.25));
        assert!(inv.approx_eq(expected), "Inverse scale should have reciprocal factors");
    }

    #[test]
    fn test_inverse_composite() {
        // Test inverse of combined transformations
        let transform = Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)) *
                       Mat4::from_rotation_y(FRAC_PI_2) *
                       Mat4::from_scale(Vec3::new(2.0, 2.0, 2.0));
        
        let inv = transform.inverse().expect("Composite matrix should be invertible");
        
        // Verify inverse works
        let point = Vec3::new(1.0, 2.0, 3.0);
        let transformed = transform.transform_point(point);
        let restored = inv.transform_point(transformed);
        assert!(restored.approx_eq(point), "Inverse should undo transformation");
    }

    #[test]
    fn test_singular_matrix() {
        // Create a singular matrix (columns are linearly dependent)
        let singular = Mat4::new(
            Vec4::new(1.0, 2.0, 3.0, 4.0),
            Vec4::new(2.0, 4.0, 6.0, 8.0),  // 2x first column
            Vec4::new(3.0, 6.0, 9.0, 12.0), // 3x first column
            Vec4::new(4.0, 8.0, 12.0, 16.0) // 4x first column
        );
        
        assert!(!singular.is_invertible(), "Singular matrix should not be invertible");
        assert!(singular.inverse().is_none(), "Inverse of singular matrix should be None");
    }

    #[test]
    fn test_inverse_properties() {
        // Test that A * A⁻¹ = I
        let a = Mat4::from_rotation_z(FRAC_PI_4) * 
                Mat4::from_scale(Vec3::new(2.0, 3.0, 1.0)) *
                Mat4::from_translation(Vec3::new(1.0, -2.0, 3.0));
        
        let a_inv = a.inverse().expect("Matrix should be invertible");
        let product = a * a_inv;
        
        assert!(product.approx_eq(Mat4::identity()), 
               "A * A⁻¹ should equal identity matrix");
    }

    #[test]
    fn test_determinant() {
        // Identity matrix has determinant 1
        assert!(epsilon_eq(Mat4::identity().determinant(), 1.0, 1e-6));
        
        // Scaling affects determinant
        let scale = Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0));
        assert!(epsilon_eq(scale.determinant(), 2.0 * 3.0 * 4.0, 1e-6));
        
        // Rotation doesn't change determinant magnitude
        let rot = Mat4::from_rotation_x(FRAC_PI_2);
        assert!(epsilon_eq(rot.determinant().abs(), 1.0, 1e-6));
        
        // Singular matrix has zero determinant
        let singular = Mat4::new(
            Vec4::new(1.0, 2.0, 3.0, 4.0),
            Vec4::new(1.0, 2.0, 3.0, 4.0),
            Vec4::new(1.0, 2.0, 3.0, 4.0),
            Vec4::new(1.0, 2.0, 3.0, 4.0)
        );
        assert!(epsilon_eq(singular.determinant(), 0.0, 1e-6));
    }
}