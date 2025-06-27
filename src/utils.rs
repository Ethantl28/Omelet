
///Returns true if 2 floats are approx equal within epsilon
pub fn epsilon_eq(a: f32, b: f32, epsilon: f32) -> bool {
    (a - b).abs() < epsilon
}

///Converts degrees to radians
pub fn degrees_to_radians(degrees: f32) -> f32 {
    degrees * std::f32::consts::PI / 180.0
}

///Converts radians to degrees
pub fn radians_to_degrees(radians: f32) -> f32 {
    radians * 180.0 / std::f32::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_degree_radian_conversion() {
        // Test common angles
        assert!(epsilon_eq(degrees_to_radians(0.0), 0.0, 1e-6));
        assert!(epsilon_eq(degrees_to_radians(90.0), PI/2.0, 1e-6));
        assert!(epsilon_eq(degrees_to_radians(180.0), PI, 1e-6));
        assert!(epsilon_eq(degrees_to_radians(360.0), 2.0*PI, 1e-6));
        
        // Test round-trip conversion
        let angle = 45.0;
        let converted = radians_to_degrees(degrees_to_radians(angle));
        assert!(epsilon_eq(angle, converted, 1e-6));
        
        let rad = PI/4.0;
        let converted = degrees_to_radians(radians_to_degrees(rad));
        assert!(epsilon_eq(rad, converted, 1e-6));
    }

    #[test]
    fn test_epsilon_eq() {
        assert!(epsilon_eq(1.0, 1.0000001, 1e-5));
        assert!(!epsilon_eq(1.0, 1.1, 1e-5));
    }
}