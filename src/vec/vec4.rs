use crate::utils;
use crate::utils::epsilon_eq;
use crate::vec::Vec3;

///A 3D vector with x, y and z components
#[derive(Debug, Clone, Copy)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    ///Creates a new Vec4
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
        Vec4{x, y, z, w}
    }

    ///Creates a vector with only zeros
    pub fn zero() -> Vec4 {
        Vec4::new(0.0, 0.0, 0.0, 0.0)
    }

    ///Returns vector with absolute of each component
    pub fn abs(self) -> Vec4 {
        Vec4::new(self.x.abs(), self.y.abs(), self.z.abs(), self.w.abs())
    }

    ///Returns a new vector with the sign of each component (-1, 0, or 1)
    pub fn signum(self) -> Vec4 {
        Vec4::new(self.x.signum(), self.y.signum(), self.z.signum(), self.w.signum())
    }

    ///Clamps each element in the vector to a min and max
    pub fn clamp(self, min: f32, max: f32) -> Vec4 {
        Vec4::new(utils::clamp(self.x, min, max), utils::clamp(self.y, min, max), utils::clamp(self.z, min, max), utils::clamp(self.w, min, max))
    }

    ///Returns array of floats
    pub fn to_array(self) -> (f32, f32, f32, f32) {
        (self.x, self.y, self.z, self.w)
    }

    ///Returns vector from array of floats
    pub fn from_array(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
        Vec4::new(x, y, z, w)
    }

    ///Extract XYZ components
    pub fn xyz(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    ///Returns the magnitude of the vector
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    ///Returns squared magnitude of vector
    pub fn squared_length(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    ///Returns the dot product of self and other
    pub fn dot(&self, other: Vec4) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    ///Returns result of cross product (WARNING: ASSUMES w IS 0)
    pub fn cross_xyz(&self, other: Vec4) -> Vec4 {
        Vec4::new(
            self.y * other.z - self.z * other.y,    //x
            self.z * other.x - self.x * other.z,    //y
            self.x * other.y - self.y * other.x,    //z
            0.0
        )
    }

    ///Returns a normalized vector
    pub fn normalize(&self) -> Vec4 {
        let len = self.length();
        if len == 0.0 {
            Vec4::new(0.0, 0.0, 0.0, 0.0)
        } else {
            Vec4::new(self.x / len, self.y / len, self.z / len, self.w / len)
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
    pub fn distance(&self, other: Vec4) -> f32 {
        (*self - other).length()
    }

    ///Returns distance between 2 vectors, avoids sqrt
    pub fn squared_distance(&self, other: Vec4) -> f32 {
        (*self - other).squared_length()
    }

    ///Returns direction to another vector
    pub fn direction_to(&self, other: Vec4) -> Vec4 {
        let delta = other - *self;
        delta.normalize()
    }

    ///Returns the raw direction vector (not normalized)
    pub fn direction_to_raw(&self, other: Vec4) -> Vec4 {
        other - *self
    }

    ///Move from current position towards target, with max distance delta
    pub fn move_towards(current: Vec4, target: Vec4, max_delta: f32) -> Vec4 {
        let delta = target - current;
        let distance = delta.length();

        if distance <= max_delta || distance < f32::EPSILON {
            target
        } else {
            current + delta / distance * max_delta
        }
    }

    ///Returns vec3 of projection of A to B
    pub fn project(&self, onto: Vec4) -> Vec4 {
        let denominator = onto.dot(onto);
        if denominator == 0.0 {
            Vec4::new(0.0, 0.0, 0.0, 0.0)
        } else {
            onto * (self.dot(onto) / denominator)
        }
    }

    ///Returns vec3 of rejection
    pub fn reject(&self, other: Vec4) -> Vec4 {
        *self - self.project(other)
    }

    ///Returns reflection (mirror over a normal)
    pub fn reflect(&self, normal: Vec4) -> Vec4 {
        *self - normal * (2.0 * self.dot(normal))
    }

    /// Checks if two vectors are approximately equal using epsilon.
    pub fn approx_eq(&self, other: Vec4) -> bool {
        epsilon_eq(self.x, other.x, 1e-6) && epsilon_eq(self.y, other.y, 1e-6)
    }

    ///Returns linear interpolation between two vectors
    pub fn lerp(a: Vec4, b: Vec4, t: f32) ->  Vec4 {
        a * (1.0 - t) + b * t
    }

    /// Spherical interpolation (works for normalized vectors)
    pub fn slerp(a: Vec4, b: Vec4, t: f32) -> Vec4 {
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
    
        if dot > 1.0 - 1e-6 {
            return Vec4::lerp(a, b, t);
        }
    
        let theta = dot.acos();
        let sin_theta = theta.sin();
        let wa = ((1.0 - t) * theta).sin() / sin_theta;
        let wb = (t * theta).sin() / sin_theta;
    
        (a_norm * wa + b_norm * wb) * (a.length() * (1.0 - t) + b.length() * t)
    }

    ///Returns true if all components are finite (not NaN or infinity)
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }

    ///Returns true if any components is NaN
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan() || self.w.is_nan()
    }
}

///Operator overloads
///Addition for Vec4
use std::ops::Add;

impl Add for Vec4 {
    type Output = Vec4;
    fn add(self, rhs: Vec4) -> Self::Output {
        Vec4::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z, self.w + rhs.w)
    }
}

///Subtraction for Vec4
use std::ops::Sub;

impl Sub for Vec4 {
    type Output = Vec4;
    fn sub(self, rhs: Vec4) -> Self::Output {
        Vec4::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z, self.w - rhs.w)
    }
}

///Multiplication for Vec4
use std::ops::Mul;

impl Mul for Vec4 {
    type Output = Vec4;
    fn mul(self, rhs: Vec4) -> Self::Output {
        Vec4::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z, self.w * rhs.w)
    }
}

///Scalar Multiplication for Vec4
impl Mul<f32> for Vec4 {
    type Output = Vec4;
    fn mul(self, scalar: f32) -> Self::Output {
        Vec4::new(self.x * scalar, self.y * scalar, self.z * scalar, self.w * scalar)
    }   
}

impl Mul<Vec4> for f32 {
    type Output = Vec4;
    fn mul(self, vec: Vec4) -> Vec4 {
        Vec4::new(vec.x * self, vec.y * self, vec.z * self, vec.w * self)
    }
}

///Divison for Vec4
use std::ops::Div;

impl Div for Vec4 {
    type Output = Vec4;
    fn div(self, rhs: Vec4) -> Self::Output {
        Vec4::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z, self.w / rhs.w)
    }
}

///Scalar division for Vec4
impl Div<f32> for Vec4 {
    type Output = Vec4;
    fn div(self, scalar: f32) -> Self::Output {
        Vec4::new(self.x / scalar, self.y / scalar, self.z / scalar, self.w / scalar)
    }
}

impl Div<Vec4> for f32 {
    type Output = Vec4;
    fn div(self, vec: Vec4) -> Vec4 {
        Vec4::new(vec.x / self, vec.y / self, vec.z / self, vec.w / self)
    }
}

///Negate Vec4
use std::ops::Neg;

impl Neg for Vec4 {
    type Output = Self;
    fn neg(self) -> Self {
        Vec4::new(-self.x, -self.y, -self.z, -self.w)
    }
}

///Equality check for Vec4
use std::cmp::PartialEq;

impl PartialEq for Vec4 {
    fn eq(&self, other: &Self) -> bool {
        self.approx_eq(*other)
    }
}

impl Default for Vec4 {
    fn default() -> Self {
        Vec4 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        }
    }
}

use std::ops::{Index, IndexMut};

impl Index<usize> for Vec4 {
    type Output = f32;
    
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Vec4 index out of bounds: {}", index),
        }
    }
}

impl IndexMut<usize> for Vec4 {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Vec4 index out of bounds: {}", index),
        }
    }
}