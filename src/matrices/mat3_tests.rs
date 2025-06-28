use crate::mat3::Mat3;
use crate::vec3::Vec3;
use crate::vec2::Vec2;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2};
    
    #[test]
    fn test_construction() {
        let m = Mat3::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0)
        );
        assert_eq!(m.x, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(m.y, Vec3::new(4.0, 5.0, 6.0));
        assert_eq!(m.z, Vec3::new(7.0, 8.0, 9.0));
    }

    #[test]
    fn test_identity() {
        let m = Mat3::identity();
        assert_eq!(m.mul_vec3(Vec3::new(1.0, 2.0, 3.0)), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_scale() {
        let scale = Mat3::from_scale(Vec3::new(2.0, 3.0, 4.0));
        assert_eq!(scale.mul_vec3(Vec3::new(1.0, 1.0, 1.0)), Vec3::new(2.0, 3.0, 4.0));
    }

    #[test]
    fn test_translation() {
        let trans = Mat3::from_translation(Vec2::new(5.0, 6.0));
        let point = trans.transform_point(Vec2::new(1.0, 2.0));
        assert_eq!(point, Vec2::new(6.0, 8.0));
        
        let vector = trans.transform_vector(Vec2::new(1.0, 2.0));
        assert_eq!(vector, Vec2::new(1.0, 2.0)); // Vectors shouldn't translate
    }

    #[test]
    fn test_rotation_z() {
        let rot = Mat3::from_rotation_z(FRAC_PI_2); // 90 degrees
        let v = rot.transform_vector(Vec2::new(1.0, 0.0));
        assert!(v.approx_eq(Vec2::new(0.0, 1.0)));
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Mat3::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0)
        );
        let b = Mat3::identity();
        assert_eq!(a * b, a);
    }

    #[test]
    fn test_determinant() {
        let m = Mat3::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 3.0)
        );
        assert_eq!(m.determinant(), 6.0);
    }

    #[test]
    fn test_inverse() {
        let m = Mat3::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 4.0)
        );
        let inv = m.inverse().unwrap();
        let ident = m * inv;
        assert!(ident.approx_eq(Mat3::identity()));
    }

    #[test]
    fn test_transform_point_vs_vector() {
        let m = Mat3::from_translation(Vec2::new(5.0, 6.0));
        let point = Vec2::new(1.0, 2.0);
        let vector = Vec2::new(1.0, 2.0);
        
        assert_eq!(m.transform_point(point), Vec2::new(6.0, 8.0));
        assert_eq!(m.transform_vector(vector), vector); // Vectors shouldn't translate
    }
}