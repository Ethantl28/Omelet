use crate::mat4::Mat4;
use crate::vec::{vec3::Vec3, vec4::Vec4};
use crate::utils::epsilon_eq;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{PI, FRAC_PI_2};

    #[test]
    fn test_identity() {
        let m = Mat4::identity();
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(m.mul_vec4(v), v);
    }

    #[test]
    fn test_translation() {
        let trans = Mat4::from_translation(Vec3::new(5.0, 6.0, 7.0));
        let point = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(trans.transform_point(point), Vec3::new(6.0, 8.0, 10.0));
        
        // Vectors shouldn't be affected by translation
        assert_eq!(trans.transform_vector(point), point);
    }

    #[test]
    fn test_rotation_x() {
        let rot = Mat4::from_rotation_x(FRAC_PI_2); // 90 degrees
        let v = Vec3::new(0.0, 1.0, 0.0);
        let result = rot.transform_vector(v);
        assert!(result.approx_eq(Vec3::new(0.0, 0.0, 1.0)));
    }

    #[test]
    fn test_perspective_projection() {
        // Test standard perspective
        let fov = PI / 2.0; // 90 degrees
        let aspect = 16.0 / 9.0;
        let near = 0.1;
        let far = 100.0;
        let proj = Mat4::from_perspective(fov, aspect, near, far);

        // Test near plane
        let near_point = Vec3::new(0.0, 0.0, -near);
        let projected_near = proj.transform_point(near_point);
        assert!(epsilon_eq(projected_near.z, -1.0, 1e-6));

        // Test far plane
        let far_point = Vec3::new(0.0, 0.0, -far);
        let projected_far = proj.transform_point(far_point);
        assert!(epsilon_eq(projected_far.z, 1.0, 1e-6));

        // Test middle point
        let mid_z = -(near + far) / 2.0;
        let mid_point = Vec3::new(0.0, 0.0, mid_z);
        let projected_mid = proj.transform_point(mid_point);
        assert!(projected_mid.z.abs() < 1.0);
    }

    #[test]
    fn test_perspective_frustum() {
        let proj = Mat4::from_perspective(FRAC_PI_2, 1.0, 1.0, 10.0);
        
        // Test frustum corners
        let right_top = proj.transform_point(Vec3::new(1.0, 1.0, -1.0));
        assert!(right_top.x > 0.0 && right_top.y > 0.0);
        
        let left_bottom = proj.transform_point(Vec3::new(-1.0, -1.0, -1.0));
        assert!(left_bottom.x < 0.0 && left_bottom.y < 0.0);
    }

    #[test]
    fn test_perspective_depth() {
        let proj = Mat4::from_perspective(FRAC_PI_2, 1.0, 0.1, 100.0);
        
        // Near plane should map to -1
        let near = proj.transform_point(Vec3::new(0.0, 0.0, -0.1));
        assert!(epsilon_eq(near.z, -1.0, 1e-6));
        
        // Far plane should map to 1
        let far = proj.transform_point(Vec3::new(0.0, 0.0, -100.0));
        assert!(epsilon_eq(far.z, 1.0, 1e-6));
        
        // Mid point should be between -1 and 1
        let mid = proj.transform_point(Vec3::new(0.0, 0.0, -50.0));
        assert!(mid.z > -1.0 && mid.z < 1.0);
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let b = Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0));
        let combined = a * b;
        
        // Translation should be unaffected by post-scale
        assert_eq!(combined.w.xyz(), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_transform_chain() {
        let scale = Mat4::from_scale(Vec3::new(2.0, 2.0, 2.0));
        let rotate = Mat4::from_rotation_y(FRAC_PI_2);
        let translate = Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0));
        
        let transform = translate * rotate * scale;
        let point = Vec3::new(1.0, 0.0, 0.0);
        
        // Should scale, then rotate (z becomes -x), then translate
        assert!(transform.transform_point(point).approx_eq(Vec3::new(5.0, 0.0, -2.0)));
    }
}