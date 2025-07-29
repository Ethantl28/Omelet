use crate::matrices::Mat3;
use crate::utils::{epsilon_eq, epsilon_eq_default};
use crate::vec::{Vec2, Vec3};
use core::f32;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

/// A 2x2 column-major matrix used for 2D linear transformations such as
/// rotation, scaling, and shearing.
///
/// The matrix is composed of two `Vec2` column vectors:
///
/// ```text
/// [col0.x, col1.x]
/// [col0.y, col1.y]
/// ```
///
/// # Example
/// ```rust
/// use omelet::vec::Vec2;
/// use omelet::matrices::Mat2;
///
/// let col0 = Vec2::new(1.0, 0.0);
/// let col1 = Vec2::new(0.0, 1.0);
/// let identity = Mat2::new(col0, col1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat2 {
    /// The first column of the matrix
    pub col0: Vec2,
    /// The second column of the matrix
    pub col1: Vec2,
}

impl Mat2 {
    // ============= Construction and Conversion =============
    /// Constructs a new 2x2 matrix from two column vectors
    ///
    /// The matrix is structured in column-major order:
    /// ```text
    /// [col0.x, col1.x]
    /// [col0.y, col1.y]
    /// ```
    pub fn new(col0: Vec2, col1: Vec2) -> Mat2 {
        Mat2 { col0, col1 }
    }

    /// Returns the 2x2 identity matrix
    ///
    /// # Example
    /// ```rust
    /// use omelet::vec::Vec2;
    /// use omelet::matrices::Mat2;
    /// let identity = Mat2::identity();
    /// assert_eq!(identity, Mat2::new(Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)));
    /// ```
    pub fn identity() -> Mat2 {
        Mat2::new(Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0))
    }

    /// Returns a matrix filled with zeroes
    ///
    /// This matrix represents a linear transformation that collapses any vector to zero
    pub fn zero() -> Mat2 {
        Mat2::new(Vec2::zero(), Vec2::zero())
    }

    /// Constructs a matrix from row vectors
    ///
    /// # Example
    /// ```rust
    /// use omelet::vec::Vec2;
    /// use omelet::matrices::Mat2;
    ///
    /// let m = Mat2::from_rows(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    /// assert_eq!(m, Mat2::new(Vec2::new(1.0, 3.0), Vec2::new(2.0, 4.0)));
    /// ```
    pub fn from_rows(r0: Vec2, r1: Vec2) -> Mat2 {
        Mat2 {
            col0: Vec2::new(r0.x, r1.x),
            col1: Vec2::new(r0.y, r1.y),
        }
    }

    /// Constructs a matrix representing a counter-clockwise rotation
    ///
    /// # Parameters
    /// - `radians`: Rotation angle in radians
    ///
    /// # Example
    /// ```rust
    /// use omelet::matrices::Mat2;
    /// let rot90 = Mat2::from_angle(std::f32::consts::FRAC_PI_2);
    /// ```
    pub fn from_angle(radians: f32) -> Mat2 {
        let (s, c) = radians.sin_cos();
        Mat2::new(Vec2::new(c, s), Vec2::new(-s, c))
    }

    /// Constructs a scale matrix from a scale vector
    ///
    /// # Example
    /// ```rust
    /// use omelet::matrices::Mat2;
    /// use omelet::vec::Vec2;
    /// let scale = Mat2::from_scale(Vec2::new(2.0, 3.0));
    /// assert_eq!(scale, Mat2::new(Vec2::new(2.0, 0.0), Vec2::new(0.0, 3.0)));
    /// ```
    pub fn from_scale(scale: Vec2) -> Mat2 {
        Mat2::new(Vec2::new(scale.x, 0.0), Vec2::new(0.0, scale.y))
    }

    /// Returns the transpose of the matrix
    ///
    /// Swaps rows and columns:
    /// ```text
    /// [a, b]  =>  [a, c]
    /// [c, d]  =>  [b, d]
    /// ```
    pub fn transpose(&self) -> Mat2 {
        Mat2::new(
            Vec2::new(self.col0.x, self.col1.x),
            Vec2::new(self.col0.y, self.col1.y),
        )
    }

    /// Returns the determinant of the matrix
    ///
    /// Used for checking invertibility or computing area under transformations
    ///
    /// ```text
    /// det = a * d- b * c
    /// ```
    pub fn determinant(&self) -> f32 {
        self.col0.x * self.col1.y - self.col0.y * self.col1.x
    }

    /// Returns the inverse of the matrix, if it exists
    ///
    /// Returns `None` if the matrix is singular (non-invertible)
    ///
    /// # Panics
    /// Will not panic. Returns `None` instead of panicking on singular matrices
    pub fn inverse(&self) -> Option<Mat2> {
        let det = self.determinant();
        if epsilon_eq_default(det.abs(), 0.0) {
            return None;
        }

        let inv_det = 1.0 / det;

        let a = self.col0.x;
        let b = self.col0.y;
        let c = self.col1.x;
        let d = self.col1.y;

        Some(Mat2::new(
            Vec2::new(d, -b) * inv_det,
            Vec2::new(-c, a) * inv_det,
        ))
    }

    /// Returns `true` if the matrix is invertible (non-zero determinant)
    pub fn is_invertible(&self) -> bool {
        let det = self.determinant();
        return det.abs() <= f32::EPSILON || !det.is_finite();
    }

    /// Returns a row major 2 dimensional array `[[f32; 2]; 2]`
    ///
    /// Equivalent to `[[col0.x, col1.x], [col0.y, col1.y]]`
    pub fn to_array_2d_row_major(&self) -> [[f32; 2]; 2] {
        [[self.col0.x, self.col1.x], [self.col0.y, self.col1.y]]
    }

    /// Returns a row major flat array `[f32; 4]`
    ///
    /// Equivalent to `[col0.x, col0.y, col1.x, col1.y]`
    pub fn to_array_row_major(&self) -> [f32; 4] {
        [self.col0.x, self.col0.y, self.col1.x, self.col1.y]
    }

    /// Returns a column major 2 dimensional array `[[f32; 2]; 2]`
    ///
    /// Equivalent to `[[col0.x, col0.y], [col1.x, col1.y]]`
    pub fn to_array_2d_col_major(&self) -> [[f32; 2]; 2] {
        [[self.col0.x, self.col0.y], [self.col1.x, self.col1.y]]
    }

    /// Returns a column major flat array `[f32; 4]`
    ///
    /// Equivalent to `[col0.x, col0.y, col1.x, col1.y]`
    pub fn to_array_col_major(&self) -> [f32; 4] {
        [self.col0.x, self.col0.y, self.col1.x, self.col1.y]
    }

    /// Returns a row major 2 dimensional tuple `((f32, f32), (f32, f32))`
    ///
    /// Equivalent to `((col0.x, col1.x), (col0.y, col1.y))`
    pub fn to_tuple_2d_row_major(&self) -> ((f32, f32), (f32, f32)) {
        ((self.col0.x, self.col1.x), (self.col0.y, self.col1.y))
    }

    /// Returns a row major tuple `(f32, f32, f32, f32)`
    ///
    /// Equivalent to `(col0.x, col0.y, col1.x, col1.y)`
    pub fn to_tuple_row_major(&self) -> (f32, f32, f32, f32) {
        (self.col0.x, self.col0.y, self.col1.x, self.col1.y)
    }

    /// Returns a column major 2 dimensional tuple `((f32, f32), (f32, f32))`
    ///
    /// Equivalent to `((col0.x, col0.y), (col1.x, col1.y))`
    pub fn to_tuple_2d_col_major(&self) -> ((f32, f32), (f32, f32)) {
        ((self.col0.x, self.col0.y), (self.col1.x, self.col1.y))
    }

    /// Returns a column major tuple `(f32, f32, f32, f32)`
    ///
    /// Equivalent to `(col0.x, col0.y, col1.x, col1.y)`
    pub fn to_tuple_col_major(&self) -> (f32, f32, f32, f32) {
        (self.col0.x, self.col0.y, self.col1.x, self.col1.y)
    }

    /// Returns a `Mat2` from a 2 dimensional array
    ///
    /// # Paramneters
    /// - `arr`: 2 dimensional array `[[f32; 2]; 2]`
    pub fn from_2d_array(arr: [[f32; 2]; 2]) -> Mat2 {
        Mat2::new(
            Vec2::new(arr[0][0], arr[0][1]),
            Vec2::new(arr[1][0], arr[1][1]),
        )
    }

    /// Returns a `Mat2` from a flat array
    ///
    /// # Parameters
    /// - `arr`: Flat array `[f32; 4]`
    ///
    /// Equivalent to `Mat2{Vec2{arr[0], arr[1]}, Vec2[arr[2], arr[3]]}}`
    pub fn from_array(arr: [f32; 4]) -> Mat2 {
        Mat2::new(Vec2::new(arr[0], arr[1]), Vec2::new(arr[2], arr[3]))
    }

    /// Returns a `Mat2` from a 2 dimensional tuple
    ///
    /// # Parameters
    /// - `t`: Tuple to use `((f32, f32), (f32, f32))`
    pub fn from_2d_tuple(t: ((f32, f32), (f32, f32))) -> Mat2 {
        Mat2::new(Vec2::new(t.0.0, t.0.1), Vec2::new(t.1.0, t.1.1))
    }

    /// Returns a `Mat2` from a tuple
    ///
    /// # Parameters
    /// - `t`: Tuple `(f32, f32, f32, f32)`
    ///
    /// Equivalent to `Mat2{Vec2{t.0, t.1}, Vec2{t.2, t.3}}`
    pub fn from_tuple(t: (f32, f32, f32, f32)) -> Mat2 {
        Mat2::new(Vec2::new(t.0, t.1), Vec2::new(t.2, t.3))
    }

    /// Returns the specified column vector.
    ///
    /// # Panics
    /// Panics if `col_idx` is not 0 or 1.
    pub fn col(&self, col_idx: usize) -> Vec2 {
        match col_idx {
            0 => Vec2::new(self.col0.x, self.col1.x),
            1 => Vec2::new(self.col0.y, self.col1.y),
            _ => panic!("Mat2 column index out of bounds: {}", col_idx),
        }
    }

    /// Returns a 3x3 column-major matrix.
    ///
    /// # Returns
    /// A `Mat3`:
    /// ```text
    /// [a, c, 0]
    /// [b, d, 0]
    /// [0, 0, 1]
    /// ```
    pub fn to_mat3(&self) -> Mat3 {
        Mat3::new(
            Vec3::new(self.col0.x, self.col0.y, 0.0),
            Vec3::new(self.col1.x, self.col1.y, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
    }

    // ============= Math Utilities =============

    /// Returns a matrix with the absolute value of each component
    ///
    /// Useful for bounding box operations or eliminating directional signs
    pub fn abs(self) -> Mat2 {
        Mat2 {
            col0: self.col0.abs(),
            col1: self.col1.abs(),
        }
    }

    /// Returns the sign (-1.0, 0.0, 1.0) of each matrix component
    ///
    /// Useful when analyzing directionality of matrix effects
    pub fn signum(self) -> Mat2 {
        Mat2 {
            col0: self.col0.signum(),
            col1: self.col1.signum(),
        }
    }

    /// Linearly interpolates between `self` and matrix `b` by amount `t`
    ///
    /// Equivalent to `self * (1.0 - t) + b * t`
    ///
    /// # Parameters
    /// - `b`: The target matrix
    /// - `t`: Interpolation factor
    ///
    /// # Example
    /// ```rust
    /// use omelet::matrices::Mat2;
    /// use omelet::vec::Vec2;
    ///
    /// let a = Mat2::identity();
    /// let b = Mat2::from_scale(Vec2::new(2.0, 2.0));
    /// let halfway = a.lerp(b, 0.5);
    /// ```
    pub fn lerp(&self, b: Mat2, t: f32) -> Mat2 {
        Mat2::new(
            self.col0.lerp(b.col0, t),
            self.col1.lerp(b.col1, t),
        )
    }

    /// Linearly interpolates between matrix `a` and matrix `b` by amount `t`
    ///
    /// # Parameters
    /// - `a`: Starting matrix
    /// - `b`: Target matrix
    /// - `t`: Interpolation factor
    ///
    /// Equivalent to `a * (1.0 - t) + b * t`
    pub fn lerp_between(a: Mat2, b: Mat2, t: f32) -> Mat2 {
        Mat2::new(
            a.col0.lerp(b.col0, t),
            a.col1.lerp(b.col1, t),
        )
    }

    /// Checks if all components in `self` are approximately equal to
    /// all components in `other` with a default epsilon
    ///
    /// # Parameters
    /// - `other`: The other matrix in the comparison
    pub fn approx_eq(&self, other: Mat2) -> bool {
        return self.col0.approx_eq(other.col0) && self.col1.approx_eq(other.col1);
    }

    /// Checks if all components in `self` are approximately equal to
    /// all components in `other` with a custom epsilon
    ///
    /// # Parameters
    /// - `other`: The other matrix in the comparison
    /// - `epsilon`: Epsilon to use in the check
    pub fn approx_eq_eps(&self, other: Mat2, epsilon: f32) -> bool {
        return self.col0.approx_eq_eps(other.col0, epsilon)
            && self.col1.approx_eq_eps(other.col1, epsilon);
    }

    /// Returns `true` if any of the components are NaN
    pub fn is_nan(&self) -> bool {
        return self.col0.is_nan() || self.col1.is_nan();
    }

    /// Returns `true` if all of the components are finite
    pub fn is_finite(&self) -> bool {
        return self.col0.is_finite() && self.col1.is_finite();
    }

    /// Computes the adjugate (classical adjoint) of the matrix.
    ///
    /// For matrix A =
    /// ```text
    /// [a, b]
    /// [c, d]
    /// ```
    /// the adjugate is:
    /// ```text
    /// [ d, -b]
    /// [-c,  a]
    /// ```
    pub fn adjugate(self) -> Mat2 {
        Mat2::new(
            Vec2::new(self.col1.y, -self.col0.y),
            Vec2::new(-self.col1.x, self.col0.x),
        )
    }

    // ============= Transform Utilities =============
    /// Extracts the scale component from the matrix by computing
    /// the length (magnitude) of each column vector
    ///
    /// This assumes that the matrix encodes a rotation+scale transform
    pub fn extract_scale(&self) -> Vec2 {
        Vec2::new(self.col0.length(), self.col1.length())
    }

    /// Returns the diagonal scale components without considering rotation
    ///
    /// This assumes the matrix is purely scaling or has scale aligned with axes
    /// It's a "raw" lookup: returns (col0.x, col1.y)
    pub fn extract_scale_raw(&self) -> Vec2 {
        Vec2::new(self.col0.x, self.col1.y)
    }

    /// Extracts the rotation angle in radians from the matrix
    ///
    /// Assumes the matrix is a pure rotation or a rotation+scale transform
    /// Uses the direction of the first column vector (X-axis) to determine angle
    pub fn extract_rotation(&self) -> f32 {
        self.col0.y.atan2(self.col0.x)
    }

    /// Returns an orthonormalized version of the matrix
    ///
    /// This ensures the columns are orthogonal unit vectors.
    /// This is useful if the matrix has accumulated numerical
    /// error or is approximately orthonormal
    ///
    /// - First column is normalized as the X-axis.
    /// - Second column is computed as the perpendicular vector (rotated 90 degrees CCW)
    pub fn orthonormalize(&self) -> Mat2 {
        let x = self.col0.normalize();
        let y = Vec2::new(-x.y, x.x);
        Mat2::new(x, y)
    }

    /// Checks if the matrix is orthogonal within a tolerance.
    ///
    /// Orthogonal matrices satisfy `M^T * M = I`.
    ///
    /// # Parameters
    /// - `epsilon`: Maximum allowed deviation from orthogonality.
    ///
    /// # Returns
    /// `true` if orthogonal within tolerance.
    pub fn is_orthogonal(&self, epsilon: f32) -> bool {
        (self.transpose() * *self).approx_eq_eps(Mat2::identity(), epsilon)
    }

    /// Returns the trace of the matrix.
    ///
    /// The trace is the sum of the diagonal elements (`a + d`).
    ///
    /// # Returns
    /// Scalar trace value.
    pub fn trace(self) -> f32 {
        self.col0.x + self.col1.y
    }

    // ============= Decomposition =============
    pub fn decompose(&self) -> (Vec2, f32) {
        (self.extract_scale(), self.extract_rotation())
    }

    // ============= Triangle Checks =============
    /// Checks if the matrix is lower-triangular within a tolerance.
    ///
    /// This means elements above the main diagonal are approx zero.
    ///
    /// # Parameters
    /// - `epsilon`: Maximum allowed magnitude of upper-triangular elements.
    ///
    /// # Returns
    /// `true` if matrix is lower-triangular within tolerance.
    pub fn is_lower_triangular(&self, epsilon: f32) -> bool {
        self.col0.y.abs() <= epsilon
    }

    /// Checks if the matrix is upper-triangular within a tolerance.
    ///
    /// This means elements below the main diagonal are approx zero.
    ///
    /// # Parameters
    /// - `epsilon`: Maximum allowed magnitude of lower-triangular elements.
    ///
    /// # Returns
    /// `true` if matrix is upper-triangular within tolerance.
    pub fn is_upper_triangular(&self, epsilon: f32) -> bool {
        self.col1.x.abs() <= epsilon
    }
}

// ============= Operator Overloads =============
/// Adds two matrices element-wise.
///
/// # Returns
/// A matrix where each element is the sum of the corresponding elements in `self` and `rhs`.
///
/// Equivalent to `self + rhs`.
impl Add for Mat2 {
    type Output = Self;
    fn add(self, rhs: Mat2) -> Mat2 {
        Mat2::new(self.col0 + rhs.col0, self.col1 + rhs.col1)
    }
}

/// Subtracts two matrices element-wise.
///
/// # Returns
/// A matrix where each element is the difference of the corresponding elements in `self` and `rhs`.
///
/// Equivalent to `self - rhs`.
impl Sub for Mat2 {
    type Output = Self;
    fn sub(self, rhs: Mat2) -> Mat2 {
        Mat2::new(self.col0 - rhs.col0, self.col1 - rhs.col1)
    }
}

/// Multiplies two matrices (matrix multiplication).
///
/// This performs standard 2D matrix multiplication: `self * rhs`.
///
/// # Returns
/// A new matrix that is the product of `self` and `rhs`.
///
/// Note: Matrices are column-major; multiplication follows: `self * rhs`.
impl Mul for Mat2 {
    type Output = Self;
    fn mul(self, rhs: Mat2) -> Mat2 {
        Mat2::new(
            // First column of result
            Vec2::new(
                self.col0.x * rhs.col0.x + self.col1.x * rhs.col0.y, // (1,1)
                self.col0.y * rhs.col0.x + self.col1.y * rhs.col0.y, // (2,1)
            ),
            // Second column of result
            Vec2::new(
                self.col0.x * rhs.col1.x + self.col1.x * rhs.col1.y, // (1,2)
                self.col0.y * rhs.col1.x + self.col1.y * rhs.col1.y, // (2,2)
            ),
        )
    }
}

/// Multiplies every element of the matrix by a scalar.
///
/// # Returns
/// A matrix scaled by `scalar`.
///
/// Equivalent to `self * scalar`.
impl Mul<f32> for Mat2 {
    type Output = Self;
    fn mul(self, scalar: f32) -> Mat2 {
        Mat2::new(self.col0 * scalar, self.col1 * scalar)
    }
}

/// Multiplies a matrix by a scalar from the left-hand side.
///
/// # Returns
/// A matrix scaled by `self` (the scalar).
///
/// Equivalent to `scalar * mat`.
impl Mul<Mat2> for f32 {
    type Output = Mat2;
    fn mul(self, mat: Mat2) -> Mat2 {
        mat * self
    }
}

/// Divides every element of the matrix by a scalar.
///
/// # Panics if `scalar` is zero.
///
/// # Returns
/// A matrix where each element is `self[i][j] / scalar`.
///
/// Equivalent to `self / scalar`.
impl Div<f32> for Mat2 {
    type Output = Self;
    fn div(self, scalar: f32) -> Self {
        assert!(scalar != 0.0, "Division by 0");
        Mat2::new(self.col0 / scalar, self.col1 / scalar)
    }
}

/// Multiplies the matrix with a vector (matrix-vector multiplication).
///
/// Treats the vector as a column vector and performs `Mat2 * Vec2`.
///
/// # Returns
/// A transformed vector resulting from the matrix-vector multiplication.
///
/// Note: This product cannot be reversed. `Vec2 * Mat2` does not work.
impl Mul<Vec2> for Mat2 {
    type Output = Vec2;

    fn mul(self, v: Vec2) -> Vec2 {
        Vec2::new(
            self.col0.x * v.x + self.col1.x * v.y,
            self.col0.y * v.x + self.col1.y * v.y,
        )
    }
}

impl approx::AbsDiffEq for Mat2 {
    type Epsilon = f32;

    fn default_epsilon() -> f32 {
        1e-6
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: f32) -> bool {
        epsilon_eq(self.col0.x, other.col0.x, epsilon)
            && epsilon_eq(self.col0.y, other.col0.y, epsilon)
            && epsilon_eq(self.col1.x, other.col1.x, epsilon)
            && epsilon_eq(self.col1.y, other.col1.y, epsilon)
    }
}

impl approx::RelativeEq for Mat2 {
    fn default_max_relative() -> f32 {
        1e-6
    }

    fn relative_eq(&self, other: &Self, epsilon: f32, max_relative: f32) -> bool {
        // You can reuse approx crate Vec2 relative_eq if available
        Vec2::relative_eq(&self.col0, &other.col0, epsilon, max_relative)
            && Vec2::relative_eq(&self.col1, &other.col1, epsilon, max_relative)
    }
}

impl Default for Mat2 {
    fn default() -> Self {
        Mat2 {
            col0: Vec2::zero(),
            col1: Vec2::zero(),
        }
    }
}

use std::ops::{Index, IndexMut};

/// Enables `m[row]` access
///
/// # Panics
/// Panics if `row >= 2`.
impl Index<usize> for Mat2 {
    type Output = Vec2;
    fn index(&self, row: usize) -> &Vec2 {
        match row {
            0 => &self.col0,
            1 => &self.col1,
            _ => panic!("Mat2 row index out of bounds: {}", row),
        }
    }
}

/// Enables mutable `m[row]` access
///
/// # Panics
/// Panics if `row >= 2`.
impl IndexMut<usize> for Mat2 {
    fn index_mut(&mut self, row: usize) -> &mut Vec2 {
        match row {
            0 => &mut self.col0,
            1 => &mut self.col1,
            _ => panic!("Mat2 row index out of bounds: {}", row),
        }
    }
}

impl fmt::Display for Mat2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print as 2 rows for readability
        writeln!(f, "[[{:.4}, {:.4}],", self.col0.x, self.col1.x)?;
        writeln!(f, "[{:.4}, {:.4}]]", self.col0.y, self.col1.y)
    }
}
