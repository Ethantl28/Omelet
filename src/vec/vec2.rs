use crate::utils;
use crate::utils::epsilon_eq;
use crate::mat2::Mat2;

///A 2D vector with x and y components
#[derive(Debug, Clone, Copy)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    ///Creates a new Vector 2
    pub const fn new(x: f32, y: f32) -> Vec2 {
        Vec2 {x, y}
    }

    ///Creates a vector with only zeros
    pub const fn zero() -> Vec2 {
        Vec2::new(0.0, 0.0)
    }

    ///Returns vector with absolute of each component
    pub fn abs(self) -> Vec2 {
        Vec2::new(self.x.abs(), self.y.abs())
    }

    ///Returns a new vector with the sign of each component (-1, 0, or 1)
    pub fn signum(self) -> Vec2 {
        Vec2::new(self.x.signum(), self.y.signum())
    }

    ///Clamps each element in the vector to a min and max
    pub fn clamp(self, min: f32, max: f32) -> Vec2 {
        Vec2::new(utils::clamp(self.x, min, max), utils::clamp(self.y, min, max))
    }

    ///Returns the magnitude of the vector
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    ///Returns array of floats
    pub fn to_array(self) -> (f32, f32) {
        (self.x, self.y)
    }

    ///Returns vector from array of floats
    pub fn from_array(x: f32, y: f32) -> Vec2 {
        Vec2::new(x, y)
    }

    ///Returns squared magnitude of vector
    pub fn squared_length(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    ///Returns the dot product of self and other
    pub fn dot(&self, other: Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    ///Returns result of cross product (float)
    pub fn cross(&self, other: Vec2) -> f32 {
        self.x * other.y - self.y * other.x
    }

    ///Returns a normalized vector
    pub fn normalize(&self) -> Vec2 {
        let len = self.length();
        if len == 0.0 {
            Vec2::new(0.0, 0.0)
        } else {
            Vec2::new(self.x / len, self.y / len)
        }
    }
    
    ///Returns bool if the vector is normalized
    pub fn is_normalized(self) -> bool {
        epsilon_eq(self.length(), 1.0, 1e-6)
    }

    pub fn is_normalized_fast(self) -> bool {
        epsilon_eq(self.squared_length(), 1.0, 1e-6)
    }

    ///Returns distance between 2 vectors
    pub fn distance(&self, other: Vec2) -> f32 {
        (*self - other).length()
    }

    ///Returns distance between 2 vectors, avoids sqrt
    pub fn squared_distance(&self, other: Vec2) -> f32 {
        (*self - other).squared_length()
    }

    ///Returns direction to another vector
    pub fn direction_to(&self, other: Vec2) -> Vec2 {
        let delta = other - *self;
        delta.normalize()
    }

    ///Returns the raw direction vector (not normalized)
    pub fn direction_to_raw(&self, other: Vec2) -> Vec2 {
        other - *self
    }

    ///Returns angle between current vector and other in radians
    pub fn angle_between_radians(&self, other: Vec2) -> f32 {
        let theta = (self.dot(other)) / (self.length() * other.length());
        utils::clamp(theta, -1.0, 1.0).acos()
    }

    ///Returns angle between current vector and other in degrees
    pub fn angle_between_degrees(&self, other: Vec2) -> f32 {
        let theta = (self.dot(other)) / (self.length() * other.length());
        utils::radians_to_degrees(utils::clamp(theta, -1.0, 1.0).acos())
    }

    ///Returns vector rotated 90 degrees
    pub fn perpendicular(self) -> Vec2 {
        let m = Mat2::from_rotation(utils::degrees_to_radians(90.0));
        m * self
    }

    ///Move from current position towards target, with max distance delta
    pub fn move_towards(current: Vec2, target: Vec2, max_delta: f32) -> Vec2 {
        let delta = target - current;
        let distance = delta.length();

        if distance <= max_delta || distance < f32::EPSILON {
            target
        } else {
            current + delta / distance * max_delta
        }
    }

    ///Returns vec2 of projection of A to B
    pub fn project(&self, onto: Vec2) -> Vec2 {
        let denominator = onto.dot(onto);
        if denominator == 0.0 {
            Vec2::new(0.0, 0.0)
        } else {
            onto * (self.dot(onto) / denominator)
        }
    }

    ///Returns vec2 of rejection
    pub fn reject(&self, other: Vec2) -> Vec2 {
        *self - self.project(other)
    }

    ///Returns reflection (mirror over a normal)
    pub fn reflect(&self, normal: Vec2) -> Vec2 {
        *self - normal * (2.0 * self.dot(normal))
    }

    /// Checks if two vectors are approximately equal using epsilon.
    pub fn approx_eq(&self, other: Vec2, epsilon: f32) -> bool {
        epsilon_eq(self.x, other.x, epsilon) && epsilon_eq(self.y, other.y, epsilon)
    }

    ///Returns linear interpolation between two vectors
    pub fn lerp(a: Vec2, b: Vec2, t: f32) ->  Vec2 {
        a * (1.0 - t) + b * t
    }

    ///Spherical linear interpolation (great arc interpolation)
    ///For 2D vectors, this simplifies to angle-based interpolation
    pub fn slerp(a: Vec2, b: Vec2, t: f32) -> Vec2 {
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
            return Vec2::lerp(a, b, t);
        }
        
        // Handle opposite directions
        if dot < -1.0 + 1e-6 {
            // Find perpendicular vector
            let perp = Vec2::new(-a_norm.y, a_norm.x);
            // Interpolate length
            let length = a.length() * (1.0 - t) + b.length() * t;
            // Rotate through perpendicular
            return if t < 0.5 {
                perp.normalize() * length
            } else {
                (-perp).normalize() * length
            };
        }

        let theta = dot.acos();
        let sin_theta = theta.sin();
        let wa = ((1.0 - t) * theta).sin() / sin_theta;
        let wb = (t * theta).sin() / sin_theta;
        
        // Interpolate both direction and magnitude
        (a_norm * wa + b_norm * wb) * (a.length() * (1.0 - t) + b.length() * t)
    }

    ///Alternative simpler 2D slerp using angle interpolation
    pub fn slerp_angle(a: Vec2, b: Vec2, t: f32) -> Vec2 {
        // Handle zero vectors
        if a.length() < 1e-6 || b.length() < 1e-6 {
            return Vec2::lerp(a, b, t);
        }

        let a_norm = a.normalize();
        let b_norm = b.normalize();
        
        // Calculate angles
        let angle_a = a_norm.y.atan2(a_norm.x);
        let angle_b = b_norm.y.atan2(b_norm.x);
        
        // Find the shortest angular distance
        let mut delta_angle = (angle_b - angle_a).rem_euclid(std::f32::consts::PI * 2.0);
        if delta_angle > std::f32::consts::PI {
            delta_angle -= std::f32::consts::PI * 2.0;
        }
        
        // Interpolate angle and length separately
        let angle = angle_a + t * delta_angle;
        let length = a.length() * (1.0 - t) + b.length() * t;
        
        Vec2::new(angle.cos(), angle.sin()) * length
    }

    ///Compute barycentric coordinates (u, v, w) for point P in triangle (A, B, C)
     pub fn barycentric(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> (f32, f32, f32) {
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
    pub fn in_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
        let (u, v, w) = Vec2::barycentric(p, a, b, c);
        u >= 0.0 && v >= 0.0 && w >= 0.0
    }

    ///Returns true if all components are finite (not NaN or infinity)
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    ///Returns true if any components is NaN
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan()
    }
}

use core::f32;
///Operator overloads
///Addition for Vec2
use std::ops::Add;

impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self.x + rhs.x, self.y + rhs.y)
    }
}

///Subtraction for Vec2
use std::ops::Sub;

impl Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self.x - rhs.x, self.y - rhs.y)
    }
}

///Multiplication for Vec2
use std::ops::Mul;

impl Mul for Vec2 {
    type Output = Vec2;
    fn mul(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self.x * rhs.x, self.y * rhs.y)
    }
}

///Scalar Multiplication for Vec2
impl Mul<f32> for Vec2 {
    type Output = Vec2;
    fn mul(self, scalar: f32) -> Self::Output {
        Vec2::new(self.x * scalar, self.y * scalar)
    }   
}

impl Mul<Vec2> for f32 {
    type Output = Vec2;
    fn mul(self, vec: Vec2) -> Vec2 {
        Vec2::new(vec.x * self, vec.y * self)
    }
}

///Divison for Vec2
use std::ops::Div;

impl Div for Vec2 {
    type Output = Vec2;
    fn div(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self.x / rhs.x, self.y / rhs.y)
    }
}

///Scalar division for Vec2
impl Div<f32> for Vec2 {
    type Output = Vec2;
    fn div(self, scalar: f32) -> Self::Output {
        Vec2::new(self.x / scalar, self.y / scalar)
    }
}

impl Div<Vec2> for f32 {
    type Output = Vec2;
    fn div(self, vec: Vec2) -> Vec2 {
        Vec2::new(vec.x / self, vec.y / self)
    }
}

///Negate Vec2
use std::ops::Neg;

impl Neg for Vec2 {
    type Output = Self;
    fn neg(self) -> Self {
        Vec2::new(-self.x, -self.y)
    }
}

///Equality check for Vec2
use std::cmp::PartialEq;

impl PartialEq for Vec2 {
    fn eq(&self, other: &Self) -> bool {
        self.approx_eq(*other, 1e-6)
    }
}

impl approx::AbsDiffEq for Vec2 {
    type Epsilon = f32;

    fn default_epsilon() -> f32 {
        f32::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: f32) -> bool {
        f32::abs_diff_eq(&self.x, &other.x, epsilon) &&
        f32::abs_diff_eq(&self.y, &other.y, epsilon)
    }
}

impl approx::RelativeEq for Vec2 {
    fn default_max_relative() -> f32 {
        f32::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: f32, max_relative: f32) -> bool {
        f32::relative_eq(&self.x, &other.x, epsilon, max_relative) &&
        f32::relative_eq(&self.y, &other.y, epsilon, max_relative)
    }
}

impl Default for Vec2 {
    fn default() -> Self {
        Vec2 {
            x: 0.0,
            y: 0.0,
        }
    }
}

use std::ops::{Index, IndexMut};

impl Index<usize> for Vec2 {
    type Output = f32;
    
    ///Enables v[index] access
    ///Panics if index >= 2
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Vec2 index out of bounds: {}", index),
        }
    }
}

impl IndexMut<usize> for Vec2 {
    ///Enables mutable v[index] access
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Vec2 index out of bounds: {}", index),
        }
    }
}