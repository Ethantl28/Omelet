use crate::vec::Vec2;
use core::f32;
use std::ops::{Add, Div, Mul, Sub};

///2x2 column-major matrix for linear transformations
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat2 {
    pub x: Vec2,
    pub y: Vec2,
}

impl Mat2 {
    ///Creates a new matrix from column vectors
    pub fn new(x: Vec2, y: Vec2) -> Mat2 {
        Mat2 { x, y }
    }

    ///Creates an identity matrix (1.0 on diagonal)
    pub fn identity() -> Mat2 {
        Mat2::new(Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0))
    }

    ///Creates a zero matrix
    pub fn zero() -> Mat2 {
        Mat2::new(Vec2::zero(), Vec2::zero())
    }

    ///Creates a diagonal matrix
    pub fn from_diagonal(diag: Vec2) -> Mat2 {
        Mat2::new(Vec2::new(diag.x, 0.0), Vec2::new(0.0, diag.y))
    }

    ///Creates a rotation matrix (angle in radians)
    pub fn from_rotation(angle_rad: f32) -> Mat2 {
        let (sin, cos) = angle_rad.sin_cos();
        Mat2::new(Vec2::new(cos, sin), Vec2::new(-sin, cos))
    }

    ///Creates a scaling matrix
    pub fn from_scale(scale: Vec2) -> Mat2 {
        Mat2::from_diagonal(scale)
    }

    ///Transposes the matrix (rows become columns)
    pub fn transpose(&self) -> Mat2 {
        Mat2::new(Vec2::new(self.x.x, self.y.x), Vec2::new(self.x.y, self.y.y))
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
            Vec2::new(-self.y.x, self.x.x) * inv_det,
        ))
    }

    ///Checks if matrices are approx equal
    pub fn approx_eq(&self, other: Mat2, epsilon: f32) -> bool {
        self.x.approx_eq(other.x, epsilon) && self.y.approx_eq(other.y, epsilon)
    }

    ///Returns true if all elements are finite (not NaN or infinity)
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    ///Returns true if any elements are NaN
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan()
    }

    ///Adjugates the current matrix (if A = [a, b \n c, d] then adjugate(A) = [d, -b \n -c, a])
    pub fn adjugate(self) -> Mat2 {
        Mat2::new(
            Vec2::new(self.y.y, -self.x.y),
            Vec2::new(-self.y.x, self.x.x),
        )
    }

    ///Returns the sum of the diagonal elements
    pub fn trace(self) -> f32 {
        self.x.x + self.y.y
    }

    ///Returns the scale component from the matrix (non-trivial)
    pub fn scale_vectors(self) -> Vec2 {
        Vec2::new(self.x.x, self.y.y)
    }

    /// Performs LU decomposition with partial pivoting
    /// Returns (P, L, U) where:
    /// - P: Permutation matrix
    /// - L: Lower triangular with unit diagonal
    /// - U: Upper triangular
    pub fn lu_decompose(&self) -> (Mat2, Mat2, Mat2) {
        let mut p = Mat2::identity();
        let mut l = Mat2::identity();
        let mut u = *self;

        // Partial pivoting
        if u.x.x.abs() < u.y.x.abs() {
            u.swap_rows(0, 1);
            p = Mat2::new(Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0));
        }

        // Compute L and U
        let factor = u.y.x / u.x.x;
        l.y.x = factor;
        u.y.x = 0.0;
        u.y.y -= factor * u.x.y;

        (p, l, u)
    }

    ///QR decomposition using Gram-Schmidt process
    /// Returns (Q, R) where:
    /// Q: orthogonal matrix
    /// R: upper triangular
    pub fn qr_decompose(&self) -> (Mat2, Mat2) {
        let a0 = self.col(0);
        let a1 = self.col(1);

        // First basis vector
        let u0 = a0;
        let e0 = u0.normalize();

        // Second basis vector
        let u1 = a1 - e0 * a1.dot(e0);
        let e1 = u1.normalize();

        // Construct Q and R
        let q = Mat2::from_cols(e0, e1);
        let r = Mat2::new(
            Vec2::new(a0.dot(e0), a1.dot(e0)),
            Vec2::new(0.0, a1.dot(e1)),
        );

        (q, r)
    }

    ///Swaps rows of matrix
    ///
    ///# Panics
    /// Panics if row_a or row_b >= 2
    pub fn swap_rows(&mut self, row_a: usize, row_b: usize) {
        assert!(row_a < 2 && row_b < 2, "Row indices must be 0 or 1");
        if row_a == row_b {
            return;
        }

        // Swap x components
        let temp = self[row_a].x;
        self[row_a].x = self[row_b].x;
        self[row_b].x = temp;

        // Swap y components
        let temp = self[row_a].y;
        self[row_a].y = self[row_b].y;
        self[row_b].y = temp;
    }

    ///Gets column by index
    pub fn col(&self, col_idx: usize) -> Vec2 {
        match col_idx {
            0 => Vec2::new(self.x.x, self.y.x),
            1 => Vec2::new(self.x.y, self.y.y),
            _ => panic!("Mat2 column index out of bounds: {}", col_idx),
        }
    }

    ///Creates matrix from column vectors
    pub fn from_cols(x: Vec2, y: Vec2) -> Self {
        Mat2::new(x, y)
    }

    ///Checks if matrux is orthogonal
    pub fn is_orthogonal(&self, epsilon: f32) -> bool {
        (self.transpose() * *self).approx_eq(Mat2::identity(), epsilon)
    }

    ///Checks if matrix is lower triangular
    pub fn is_lower_triangular(&self, epsilon: f32) -> bool {
        self.x.y.abs() <= epsilon
    }

    ///Checks if matrix is upper triangular
    pub fn is_upper_triangular(&self, epsilon: f32) -> bool {
        self.y.x.abs() <= epsilon
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
                self.x.x * rhs.x.x + self.y.x * rhs.x.y, // (1,1)
                self.x.y * rhs.x.x + self.y.y * rhs.x.y, // (2,1)
            ),
            // Second column of result
            Vec2::new(
                self.x.x * rhs.y.x + self.y.x * rhs.y.y, // (1,2)
                self.x.y * rhs.y.x + self.y.y * rhs.y.y, // (2,2)
            ),
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
        assert!(scalar != 0.0, "Division by 0");
        Mat2::new(self.x / scalar, self.y / scalar)
    }
}

impl Mul<Vec2> for Mat2 {
    type Output = Vec2;
    fn mul(self, v: Vec2) -> Vec2 {
        Vec2::new(
            self.x.x * v.x + self.y.x * v.y,
            self.x.y * v.x + self.y.y * v.y,
        )
    }
}

impl approx::AbsDiffEq for Mat2 {
    type Epsilon = f32;

    fn default_epsilon() -> f32 {
        f32::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: f32) -> bool {
        Vec2::abs_diff_eq(&self.x, &other.x, epsilon)
            && Vec2::abs_diff_eq(&self.y, &other.y, epsilon)
    }
}

impl approx::RelativeEq for Mat2 {
    fn default_max_relative() -> f32 {
        f32::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: f32, max_relative: f32) -> bool {
        Vec2::relative_eq(&self.x, &other.x, epsilon, max_relative)
            && Vec2::relative_eq(&self.y, &other.y, epsilon, max_relative)
    }
}

impl Default for Mat2 {
    fn default() -> Self {
        Mat2 {
            x: Vec2::zero(),
            y: Vec2::zero(),
        }
    }
}

use std::ops::{Index, IndexMut};

impl Index<usize> for Mat2 {
    type Output = Vec2;

    ///Enables m[row] access
    ///Panics if row >= 2
    fn index(&self, row: usize) -> &Vec2 {
        match row {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Mat2 row index out of bounds: {}", row),
        }
    }
}

impl IndexMut<usize> for Mat2 {
    ///Enables mutable m[row] access
    fn index_mut(&mut self, row: usize) -> &mut Vec2 {
        match row {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Mat2 row index out of bounds: {}", row),
        }
    }
}
