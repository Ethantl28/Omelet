use crate::utils;
use crate::utils::epsilon_eq;

///A 3D vector with x, y and z components
#[derive(Debug, Clone, Copy)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    ///Creates a new Vec3
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3{x, y, z}
    }

    ///Creates a vector with only zeros
    pub fn zero() -> Vec3 {
        Vec3::new(0.0, 0.0, 0.0)
    }

    ///Returns vector with absolute of each component
    pub fn abs(self) -> Vec3 {
        Vec3::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    ///Returns a new vector with the sign of each component (-1, 0, or 1)
    pub fn signum(self) -> Vec3 {
        Vec3::new(self.x.signum(), self.y.signum(), self.z.signum())
    }

    ///Clamps each element in the vector to a min and max
    pub fn clamp(self, min: f32, max: f32) -> Vec3 {
        Vec3::new(utils::clamp(self.x, min, max), utils::clamp(self.y, min, max), utils::clamp(self.z, min, max))
    }

    ///Returns array of floats
    pub fn to_array(self) -> (f32, f32, f32) {
        (self.x, self.y, self.z)
    }

    ///Returns vector from array of floats
    pub fn from_array(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }
    
    ///Returns the magnitude of the vector
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    ///Returns squared magnitude of vector
    pub fn squared_length(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    ///Returns the dot product of self and other
    pub fn dot(&self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    ///Returns result of cross product (float)
    pub fn cross(&self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,    //x
            self.z * other.x - self.x * other.z,    //y
            self.x * other.y - self.y * other.x     //z
        )
    }

    ///Returns a normalized vector
    pub fn normalize(&self) -> Vec3 {
        let len = self.length();
        if len == 0.0 {
            Vec3::new(0.0, 0.0, 0.0)
        } else {
            Vec3::new(self.x / len, self.y / len, self.z / len)
        }
    }
    
    ///Returns bool if the vector is normalized
    pub fn is_normalized(self) -> bool {
        epsilon_eq(self.length(), 1.0, 1e-6)
    }

    ///Returns bool if vector is normalized (using squared_length())
    pub fn is_normalized_fast(self) -> bool {
        epsilon_eq(self.squared_length(), 1.0, 1e-6)
    }

    ///Returns distance between 2 vectors
    pub fn distance(&self, other: Vec3) -> f32 {
        (*self - other).length()
    }

    ///Returns distance between 2 vectors, avoids sqrt
    pub fn squared_distance(&self, other: Vec3) -> f32 {
        (*self - other).squared_length()
    }

    ///Returns direction to another vector
    pub fn direction_to(&self, other: Vec3) -> Vec3 {
        let delta = other - *self;
        delta.normalize()
    }

    ///Returns the raw direction vector (not normalized)
    pub fn direction_to_raw(&self, other: Vec3) -> Vec3 {
        other - *self
    }

    ///Move from current position towards target, with max distance delta
    pub fn move_towards(current: Vec3, target: Vec3, max_delta: f32) -> Vec3 {
        let delta = target - current;
        let distance = delta.length();

        if distance <= max_delta || distance < f32::EPSILON {
            target
        } else {
            current + delta / distance * max_delta
        }
    }

    ///Returns vec3 of projection of A to B
    pub fn project(&self, onto: Vec3) -> Vec3 {
        let denominator = onto.dot(onto);
        if denominator == 0.0 {
            Vec3::new(0.0, 0.0, 0.0)
        } else {
            onto * (self.dot(onto) / denominator)
        }
    }

    ///Returns vec3 of rejection
    pub fn reject(&self, other: Vec3) -> Vec3 {
        *self - self.project(other)
    }

    ///Returns reflection (mirror over a normal)
    pub fn reflect(&self, normal: Vec3) -> Vec3 {
        *self - normal * (2.0 * self.dot(normal))
    }

    /// Checks if two vectors are approximately equal using epsilon.
    pub fn approx_eq(&self, other: Vec3) -> bool {
        epsilon_eq(self.x, other.x, 1e-6) && epsilon_eq(self.y, other.y, 1e-6)
    }

    ///Checks if two vectors are qpprox equal using user entered epsilon
    pub fn approx_eq_eps(&self, other: Vec3, epsilon: f32) -> bool {
        epsilon_eq(self.x, other.x, epsilon) && epsilon_eq(self.y, other.y, epsilon)
    }

    ///Returns linear interpolation between two vectors
    pub fn lerp(a: Vec3, b: Vec3, t: f32) ->  Vec3 {
        a * (1.0 - t) + b * t
    }

    ///Spherical linear interpolation (great arc interpolation)
    pub fn slerp(a: Vec3, b: Vec3, t: f32) -> Vec3 {
        // Handle zero vectors
        if a.length() < 1e-6 {
            return b * t;
        }
        if b.length() < 1e-6 {
            return a * (1.0 - t);
        }

        let a_norm = a.normalize();
        let b_norm = b.normalize();
        let dot = a_norm.dot(b_norm).clamp(-1.0, 1.0);
    
        // Handle nearly parallel vectors
        if dot > 1.0 - 1e-6 {
            return Vec3::lerp(a, b, t);
        }
    
        let theta = dot.acos();
        let sin_theta = theta.sin();
        let wa = ((1.0 - t) * theta).sin() / sin_theta;
        let wb = (t * theta).sin() / sin_theta;
    
        // Interpolate both direction and magnitude
        (a_norm * wa + b_norm * wb) * (a.length() * (1.0 - t) + b.length() * t)
    }

    ///Alternative simpler slerp using angle interpolation (for 3D vectors, this uses the shortest rotation)
        pub fn slerp_angle(a: Vec3, b: Vec3, t: f32) -> Vec3 {
        // Handle zero vectors
        if a.length() < 1e-6 || b.length() < 1e-6 {
            return Vec3::lerp(a, b, t);
        }

        let a_norm = a.normalize();
        let b_norm = b.normalize();
        let dot = a_norm.dot(b_norm).clamp(-1.0, 1.0);
    
        // If the vectors are nearly parallel, use linear interpolation
        if dot > 1.0 - 1e-6 {
            return Vec3::lerp(a, b, t);
        }
    
        // Calculate the angle between the vectors
        let theta = dot.acos();
    
        // Interpolate using the slerp formula
        let sin_theta = theta.sin();
        let wa = ((1.0 - t) * theta).sin() / sin_theta;
        let wb = (t * theta).sin() / sin_theta;
    
        // Interpolate both direction and magnitude
        (a_norm * wa + b_norm * wb) * (a.length() * (1.0 - t) + b.length() * t)
    }

    ///Compute barycentric coordinates (u, v, w) for point P in triangle (A, B, C)
     pub fn barycentric(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (f32, f32, f32) {
        let v0 = b - a;
        let v1 = c - a;
        let v2 = p - a;
        
        let d00 = v0.dot(v0);
        let d01 = v0.dot(v1);
        let d11 = v1.dot(v1);
        let d20 = v2.dot(v0);
        let d21 = v2.dot(v1);
        
        let denom = d00 * d11 - d01 * d01;
        let v = (d11 * d20 - d01 * d21) / denom;
        let w = (d00 * d21 - d01 * d20) / denom;
        let u = 1.0 - v - w;
        
        (u, v, w)
    }

    /// Check if point is inside triangle using barycentric coordinates
    pub fn in_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
        let (u, v, w) = Vec3::barycentric(p, a, b, c);
        u >= 0.0 && v >= 0.0 && w >= 0.0
    }

    ///Returns true if all components are finite (not NaN or infinity)
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    ///Returns true if any components is NaN
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }
}

///Operator overloads
///Addition for Vec3
use std::ops::Add;

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Self::Output {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

///Subtraction for Vec3
use std::ops::Sub;

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Self::Output {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

///Multiplication for Vec3
use std::ops::Mul;

impl Mul for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

///Scalar Multiplication for Vec3
impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, scalar: f32) -> Self::Output {
        Vec3::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }   
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, vec: Vec3) -> Vec3 {
        Vec3::new(vec.x * self, vec.y * self, vec.z * self)
    }
}

///Divison for Vec3
use std::ops::Div;

impl Div for Vec3 {
    type Output = Vec3;
    fn div(self, rhs: Vec3) -> Self::Output {
        Vec3::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

///Scalar division for Vec3
impl Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, scalar: f32) -> Self::Output {
        Vec3::new(self.x / scalar, self.y / scalar, self.z / scalar)
    }
}

impl Div<Vec3> for f32 {
    type Output = Vec3;
    fn div(self, vec: Vec3) -> Vec3 {
        Vec3::new(vec.x / self, vec.y / self, vec.z / self)
    }
}

///Negate Vec3
use std::ops::Neg;

impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

///Equality check for Vec3
use std::cmp::PartialEq;

impl PartialEq for Vec3 {
    fn eq(&self, other: &Self) -> bool {
        self.approx_eq(*other)
    }
}

impl Default for Vec3 {
    fn default() -> Self {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

use std::ops::{Index, IndexMut};

impl Index<usize> for Vec3 {
    type Output = f32;
    
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 index out of bounds: {}", index),
        }
    }
}

impl IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Vec3 index out of bounds: {}", index),
        }
    }
}