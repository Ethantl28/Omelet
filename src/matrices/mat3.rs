use crate::vec::Vec3;
use crate::vec::Vec2;
use std::ops::{Add, Sub, Mul, Div};

///2x2 column-major matrix for linear transformations
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub x: Vec3,
    pub y: Vec3,
    pub z: Vec3,
}

impl Mat3 {
    ///Creates a new matrix from column vectors
    pub fn new(x: Vec3, y: Vec3, z: Vec3) -> Mat3 {
        Mat3{x, y, z}
    }

    ///Creates an identity matrix (1.0 on diagonal)
    pub fn identity() -> Mat3 {
        Mat3::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0)
        )
    }

    ///Creates a zero matrix
    pub fn zero() -> Mat3 {
        Mat3::new(Vec3::zero(), Vec3::zero(), Vec3::zero())
    }

    ///Creates a diagonal matrix
    pub fn from_diagonal(diag: Vec2) -> Mat3 {
        Mat3::new(
            Vec3::new(diag.x, 0.0, 0.0),
            Vec3::new(0.0, diag.y, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
    }

    ///Creates a translation matrix
    pub fn from_translation(translation: Vec2) -> Mat3 {
        Mat3::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(translation.x, translation.y, 1.0)
        )
    }

    ///Creates a rotation matrix (angle in radians)
    pub fn from_rotation_z(angle_rad: f32) -> Mat3 {
        let (sin, cos) = angle_rad.sin_cos();
        Mat3::new(
            Vec3::new(cos, sin, 0.0),
            Vec3::new(-sin, cos, 0.0),
            Vec3::new(0.0, 0.0, 1.0)
        )
    }

    ///Creates a scaling matrix
    pub fn from_scale(scale: Vec2) -> Mat3 {
        Mat3::from_diagonal(scale)
    }

    ///Transposes the matrix (rows become columns)
    pub fn transpose(&self) -> Mat3 {
        Mat3::new(
            Vec3::new(self.x.x, self.y.x, self.z.x),
            Vec3::new(self.x.y, self.y.y, self.z.y),
            Vec3::new(self.x.z, self.y.z, self.z.z)
        )
    }

    ///Calculates the determinant
    pub fn determinant(&self) -> f32 {
        self.x.x * (self.y.y * self.z.z - self.z.y * self.y.z)
        - self.y.x * (self.x.y * self.z.z - self.z.y * self.x.z)
        + self.z.x * (self.x.y * self.y.z - self.y.y * self.x.z)
    }

    ///Computes the inverse of the matrix if possible
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() <= f32::EPSILON {
            return None;
        }
        let inv_det = 1.0 / det;
        
        Some(Self::new(
            Vec3::new(
                (self.y.y * self.z.z - self.z.y * self.y.z) * inv_det,
                (self.z.y * self.x.z - self.x.y * self.z.z) * inv_det,
                (self.x.y * self.y.z - self.y.y * self.x.z) * inv_det,
            ),
            Vec3::new(
                (self.z.x * self.y.z - self.y.x * self.z.z) * inv_det,
                (self.x.x * self.z.z - self.z.x * self.x.z) * inv_det,
                (self.y.x * self.x.z - self.x.x * self.y.z) * inv_det,
            ),
            Vec3::new(
                (self.y.x * self.z.y - self.z.x * self.y.y) * inv_det,
                (self.z.x * self.x.y - self.x.x * self.z.y) * inv_det,
                (self.x.x * self.y.y - self.y.x * self.x.y) * inv_det,
            )
        ))
    }

    ///Multiplies matrix with vector
    pub fn mul_vec3(&self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.x.x * v.x + self.y.x * v.y + self.z.x * v.z,
            self.x.y * v.x + self.y.y * v.y + self.z.y * v.z,
            self.x.z * v.x + self.y.z * v.y + self.z.z * v.z
        )
    }

    ///Checks if matrices are approx equal
    pub fn approx_eq(&self, other: Mat3) -> bool {
        self.x.approx_eq(other.x) && self.y.approx_eq(other.y) && self.z.approx_eq(other.z)
    }

    ///Transforms point
    pub fn transform_point(&self, point: Vec2) -> Vec2 {
        let v = self.mul_vec3(Vec3::new(point.x, point.y, 1.0));
        Vec2::new(v.x, v.y)
    }

    ///Transforms vector
    pub fn transform_vector(&self, vector: Vec2) -> Vec2 {
        let v = self.mul_vec3(Vec3::new(vector.x, vector.y, 0.0));
        Vec2::new(v.x, v.y)
    }
}

//Operator overloads
impl Add for Mat3 {
    type Output = Self;
    fn add(self, rhs: Mat3) -> Mat3 {
        Mat3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Mat3 {
    type Output = Self;
    fn sub(self, rhs: Mat3) -> Mat3 {
        Mat3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul for Mat3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.mul_vec3(rhs.x),
            self.mul_vec3(rhs.y),
            self.mul_vec3(rhs.z)
        )
    }
}

impl Mul<f32> for Mat3 {
    type Output = Self;
    fn mul(self, scalar: f32) -> Self {
        Self::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }
}


impl Mul<Mat3> for f32 {
    type Output = Mat3;
    fn mul(self, mat: Mat3) -> Mat3 {
        mat * self
    }
}

impl Div<f32> for Mat3 {
    type Output = Self;
    fn div(self, scalar: f32) -> Self {
        Mat3::new(self.x / scalar, self.y / scalar, self.z / scalar)
    }
}