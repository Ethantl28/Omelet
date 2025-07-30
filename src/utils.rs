use crate::vec3::Vec3;

///Returns true if 2 floats are approx equal within epsilon
pub fn epsilon_eq(a: f32, b: f32, epsilon: f32) -> bool {
    (a - b).abs() < epsilon
}

///Returns true if 2 floats are approx equal within epsilon (default: 1e-6)
pub fn epsilon_eq_default(a: f32, b: f32) -> bool {
    epsilon_eq(a, b, 1e-6)
}

///Converts degrees to radians
pub fn degrees_to_radians(degrees: f32) -> f32 {
    degrees * std::f32::consts::PI / 180.0
}

///Converts radians to degrees
pub fn radians_to_degrees(radians: f32) -> f32 {
    radians * 180.0 / std::f32::consts::PI
}

///Returns a clamped float value
pub fn clamp(val: f32, min: f32, max: f32) -> f32 {
    val.max(min).min(max)
}

///Lerps between 2 floats
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

///Returns true if 2 floats are approx equal to 0 within given epsilon
pub fn is_near_zero(val: f32, epsilon: f32) -> bool {
    val.abs() < epsilon
}

///Returns truw if 2 floats are approx equal to 0 within epsilon (default: 1e-6)
pub fn is_near_zero_default(val: f32) -> bool {
    val.abs() < 1e-6
}

///Gran-Schmidt process to orthonormalize a set of vectors
pub fn gram_schmidt(v1: &mut Vec3, v2: &mut Vec3, v3: &mut Vec3) {
    *v1 = v1.normalize();
    *v2 = (*v2 - *v1 * v2.dot(*v1)).normalize();
    *v3 = (*v3 - *v1 * v3.dot(*v1) - *v2 * v3.dot(*v2)).normalize();
}

///Returns an orthonormalized copy of 3 vectors
pub fn orthonormal_basis(v1: Vec3, v2: Vec3, v3: Vec3) -> (Vec3, Vec3, Vec3) {
    let mut u1 = v1;
    let mut u2 = v2;
    let mut u3 = v3;
    gram_schmidt(&mut u1, &mut u2, &mut u3);
    (u1, u2, u3)
}

pub fn are_orthonormal(v1: Vec3, v2: Vec3, v3: Vec3, epsilon: f32) -> bool {
    let dot = |a: Vec3, b: Vec3| a.dot(b);
    let norm = |v: Vec3| epsilon_eq(v.length(), 1.0, epsilon);

    norm(v1)
        && norm(v2)
        && norm(v3)
        && epsilon_eq(dot(v1, v2), 0.0, epsilon)
        && epsilon_eq(dot(v1, v3), 0.0, epsilon)
        && epsilon_eq(dot(v2, v3), 0.0, epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::{consts::PI};

    #[test]
    fn test_degree_radian_conversion() {
        // Common angles
        assert!(epsilon_eq(degrees_to_radians(0.0), 0.0, 1e-6));
        assert!(epsilon_eq(degrees_to_radians(90.0), PI / 2.0, 1e-6));
        assert!(epsilon_eq(degrees_to_radians(180.0), PI, 1e-6));
        assert!(epsilon_eq(degrees_to_radians(360.0), 2.0 * PI, 1e-6));

        // Round-trip conversion
        let angle = 45.0;
        let converted = radians_to_degrees(degrees_to_radians(angle));
        assert!(epsilon_eq(angle, converted, 1e-6));

        let rad = PI / 4.0;
        let converted = degrees_to_radians(radians_to_degrees(rad));
        assert!(epsilon_eq(rad, converted, 1e-6));
    }

    #[test]
    fn test_epsilon_eq() {
        assert!(epsilon_eq(1.0, 1.0000001, 1e-5));
        assert!(!epsilon_eq(1.0, 1.1, 1e-5));

        // Edge cases
        assert!(epsilon_eq(0.0, 1e-7, 1e-6));
        assert!(!epsilon_eq(0.0, 1e-4, 1e-6));
        assert!(!epsilon_eq(f32::NAN, f32::NAN, 1e-6));
        assert!(!epsilon_eq(f32::INFINITY, f32::INFINITY, 1e-6)); // not equal in Rust float ops
    }

    #[test]
    fn test_epsilon_eq_default() {
        assert!(epsilon_eq_default(1.0, 1.0000005));
        assert!(!epsilon_eq_default(1.0, 1.01));
    }

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5.0, 1.0, 10.0), 5.0);
        assert_eq!(clamp(-5.0, 0.0, 1.0), 0.0);
        assert_eq!(clamp(100.0, 0.0, 50.0), 50.0);
        assert_eq!(clamp(10.0, 10.0, 10.0), 10.0); // degenerate range
    }

    #[test]
    fn test_lerp() {
        assert!(epsilon_eq(lerp(0.0, 10.0, 0.0), 0.0, 1e-6));
        assert!(epsilon_eq(lerp(0.0, 10.0, 1.0), 10.0, 1e-6));
        assert!(epsilon_eq(lerp(0.0, 10.0, 0.5), 5.0, 1e-6));
        assert!(epsilon_eq(lerp(-10.0, 10.0, 0.75), 5.0, 1e-6));
    }

    #[test]
    fn test_is_near_zero() {
        assert!(is_near_zero(0.0000001, 1e-6));
        assert!(!is_near_zero(0.001, 1e-6));
    }

    #[test]
    fn test_is_near_zero_default() {
        assert!(is_near_zero_default(1e-7));
        assert!(!is_near_zero_default(1e-3));
    }

    #[test]
    fn test_gram_schmidt() {
        let mut a = Vec3::new(1.0, 2.0, 3.0);
        let mut b = Vec3::new(4.0, 5.0, 6.0);
        let mut c = Vec3::new(7.0, 8.0, 10.0);
        gram_schmidt(&mut a, &mut b, &mut c);

        assert!(are_orthonormal(a, b, c, 1e-5));
    }

    #[test]
    fn test_orthonormal_basis() {
        let a = Vec3::new(2.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 3.0, 0.0);
        let c = Vec3::new(0.0, 0.0, 4.0);

        let (o1, o2, o3) = orthonormal_basis(a, b, c);
        assert!(are_orthonormal(o1, o2, o3, 1e-5));
    }

    #[test]
    fn test_orthonormal_basis_nontrivial() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.0, 6.0);
        let v3 = Vec3::new(7.0, 8.0, 10.0); // Slightly altered to avoid perfect linear dependence

        let (u1, u2, u3) = orthonormal_basis(v1, v2, v3);

        assert!(are_orthonormal(u1, u2, u3, 1e-5));
    }

    #[test]
    fn test_gram_schmidt_idempotent() {
        // If vectors are already orthonormal, they should remain unchanged
        let mut a = Vec3::new(1.0, 0.0, 0.0);
        let mut b = Vec3::new(0.0, 1.0, 0.0);
        let mut c = Vec3::new(0.0, 0.0, 1.0);
        let original = (a, b, c);
        gram_schmidt(&mut a, &mut b, &mut c);
        assert!(are_orthonormal(a, b, c, 1e-6));
        assert!(epsilon_eq_default(a.dot(original.0), 1.0));
        assert!(epsilon_eq_default(b.dot(original.1), 1.0));
        assert!(epsilon_eq_default(c.dot(original.2), 1.0));
    }
}
