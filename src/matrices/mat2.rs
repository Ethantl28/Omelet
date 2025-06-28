use crate::vec::Vec2;
use std::ops::{Add, Sub, Mul, Div};

///2x2 column-major matrix for linear transformations
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat2 {
    pub x: Vec2,
    pub y: Vec2,
}

impl Mat2 {
    ///Creates a new matrix from column vectors
    pub fn new(x: Vec2, y: Vec2) -> Mat2 {
        Mat2{x, y}
    }

    ///Creates an identity matrix (1.0 on diagonal)
    pub fn identity() -> Mat2 {
        Mat2::new(
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0)
        )
    }

    ///Creates a zero matrix
    pub fn zero() -> Mat2 {
        Mat2::new(Vec2::zero(), Vec2::zero())
    }

    ///Creates a diagonal matrix
    pub fn from_diagonal(diag: Vec2) -> Mat2 {
        Mat2::new(
            Vec2::new(diag.x, 0.0),
            Vec2::new(0.0, diag.y)
        )
    }

    ///Creates a rotation matrix (angle in radians)
    pub fn from_rotation(angle_rad: f32) -> Mat2 {
        let (sin, cos) = angle_rad.sin_cos();
        Mat2::new(
            Vec2::new(cos, sin),
            Vec2::new(-sin, cos)
        )
    }

    ///Creates a scaling matrix
    pub fn from_scale(scale: Vec2) -> Mat2 {
        Mat2::from_diagonal(scale)
    }

    ///Transposes the matrix (rows become columns)
    pub fn transpose(&self) -> Mat2 {
        Mat2::new(
            Vec2::new(self.x.x, self.y.x),
            Vec2::new(self.x.y, self.y.y)
        )
    }

    ///Calculates the determinant
    pub fn determinant(&self) -> f32 {
        self.x.x * self.y.y - self.x.y * self.y.x
    }

    ///Computes the inverse of the matrix if possible
    pub fn inverse(&self) -> Option<Mat2> {
    let det = self.determinant();
    if det.abs() <= f32::EPSILON {
        return None;
    }
    let inv_det = 1.0 / det;
    Some(Mat2::new(
        Vec2::new(self.y.y, -self.x.y) * inv_det,
        Vec2::new(-self.y.x, self.x.x) * inv_det
    ))
}

    ///Multiplies matrix with vector
    pub fn mul_vec2(&self, v: Vec2) -> Vec2 {
        Vec2::new(
            self.x.x * v.x + self.y.x * v.y,
            self.x.y * v.x + self.y.y * v.y
        )
    }

    ///Checks if matrices are approx equal
    pub fn approx_eq(&self, other: Mat2) -> bool {
        self.x.approx_eq(other.x) && self.y.approx_eq(other.y)
    }
}

//Operator overloads
impl Add for Mat2 {
    type Output = Self;
    fn add(self, rhs: Mat2) -> Mat2 {
        Mat2::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Sub for Mat2 {
    type Output = Self;
    fn sub(self, rhs: Mat2) -> Mat2 {
        Mat2::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Mul for Mat2 {
    type Output = Self;
    fn mul(self, rhs: Mat2) -> Mat2 {
        Mat2::new(
            // First column of result
            Vec2::new(
                self.x.x * rhs.x.x + self.y.x * rhs.x.y,  // (1,1)
                self.x.y * rhs.x.x + self.y.y * rhs.x.y   // (2,1)
            ),
            // Second column of result
            Vec2::new(
                self.x.x * rhs.y.x + self.y.x * rhs.y.y,  // (1,2)
                self.x.y * rhs.y.x + self.y.y * rhs.y.y   // (2,2)
            )
        )
    }
}

impl Mul<f32> for Mat2 {
    type Output = Self;
    fn mul(self, scalar: f32) -> Mat2 {
        Mat2::new(self.x * scalar, self.y * scalar)
    }
}


impl Mul<Mat2> for f32 {
    type Output = Mat2;
    fn mul(self, mat: Mat2) -> Mat2 {
        mat * self
    }
}

impl Div<f32> for Mat2 {
    type Output = Self;
    fn div(self, scalar: f32) -> Self {
        Mat2::new(self.x / scalar, self.y / scalar)
    }
}