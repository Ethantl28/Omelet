use crate::matrices::Mat3;
use crate::utils::epsilon_eq_default;
use crate::vec::{Vec2, Vec3};
use core::f32;
use std::{
    cmp::PartialEq,
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

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
#[derive(Debug, Clone, Copy)]
pub struct Mat2 {
    /// The first column of the matrix
    pub col0: Vec2,
    /// The second column of the matrix
    pub col1: Vec2,
}

// ============= Types ==============
pub type Mat2Tuple2D = ((f32, f32), (f32, f32));
pub type Mat2Tuple = (f32, f32, f32, f32);

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

    // ============= Constants ==============
    pub const ZERO: Self = Self {
        col0: Vec2::ZERO,
        col1: Vec2::ZERO,
    };

    pub const IDENTITY: Self = Self {
        col0: Vec2 { x: 1.0, y: 0.0 },
        col1: Vec2 { x: 0.0, y: 1.0 },
    };

    pub const NAN: Self = Self {
        col0: Vec2::NAN,
        col1: Vec2::NAN,
    };

    pub const INFINITY: Self = Self {
        col0: Vec2::INFINITY,
        col1: Vec2::INFINITY,
    };

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
        self.determinant().abs() > 1e-6
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
    pub fn to_tuple_2d_row_major(&self) -> Mat2Tuple2D {
        ((self.col0.x, self.col1.x), (self.col0.y, self.col1.y))
    }

    /// Returns a row major tuple `(f32, f32, f32, f32)`
    ///
    /// Equivalent to `(col0.x, col0.y, col1.x, col1.y)`
    pub fn to_tuple_row_major(&self) -> Mat2Tuple {
        (self.col0.x, self.col0.y, self.col1.x, self.col1.y)
    }

    /// Returns a column major 2 dimensional tuple `((f32, f32), (f32, f32))`
    ///
    /// Equivalent to `((col0.x, col0.y), (col1.x, col1.y))`
    pub fn to_tuple_2d_col_major(&self) -> Mat2Tuple2D {
        ((self.col0.x, self.col0.y), (self.col1.x, self.col1.y))
    }

    /// Returns a column major tuple `(f32, f32, f32, f32)`
    ///
    /// Equivalent to `(col0.x, col0.y, col1.x, col1.y)`
    pub fn to_tuple_col_major(&self) -> Mat2Tuple {
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
    pub fn from_2d_tuple(t: Mat2Tuple2D) -> Mat2 {
        Mat2::new(Vec2::new(t.0 .0, t.0 .1), Vec2::new(t.1 .0, t.1 .1))
    }

    /// Returns a `Mat2` from a tuple
    ///
    /// # Parameters
    /// - `t`: Tuple `(f32, f32, f32, f32)`
    ///
    /// Equivalent to `Mat2{Vec2{t.0, t.1}, Vec2{t.2, t.3}}`
    pub fn from_tuple(t: Mat2Tuple) -> Mat2 {
        Mat2::new(Vec2::new(t.0, t.1), Vec2::new(t.2, t.3))
    }

    /// Returns the specified column vector.
    ///
    /// # Panics
    /// Panics if `col_idx` is not 0 or 1.
    pub fn col(&self, index: usize) -> Vec2 {
        match index {
            0 => self.col0,
            1 => self.col1,
            _ => panic!("Mat2 column index out of bounds: {}", index),
        }
    }

    pub fn row(&self, index: usize) -> Vec2 {
        match index {
            0 => Vec2::new(self.col0.x, self.col1.x),
            1 => Vec2::new(self.col0.y, self.col1.y),
            _ => panic!("Mat2 row index out of bounds: {}", index),
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
    /// let a = Mat2::IDENTITY;
    /// let b = Mat2::from_scale(Vec2::new(2.0, 2.0));
    /// let halfway = a.lerp(b, 0.5);
    /// ```
    pub fn lerp(&self, b: Mat2, t: f32) -> Mat2 {
        Mat2::new(self.col0.lerp(b.col0, t), self.col1.lerp(b.col1, t))
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
        Mat2::new(a.col0.lerp(b.col0, t), a.col1.lerp(b.col1, t))
    }

    /// Checks if all components in `self` are approximately equal to
    /// all components in `other` with a default epsilon
    ///
    /// # Parameters
    /// - `other`: The other matrix in the comparison
    pub fn approx_eq(&self, other: Mat2) -> bool {
        self.col0.approx_eq(other.col0) && self.col1.approx_eq(other.col1)
    }

    /// Checks if all components in `self` are approximately equal to
    /// all components in `other` with a custom epsilon
    ///
    /// # Parameters
    /// - `other`: The other matrix in the comparison
    /// - `epsilon`: Epsilon to use in the check
    pub fn approx_eq_eps(&self, other: Mat2, epsilon: f32) -> bool {
        self.col0.approx_eq_eps(other.col0, epsilon) && self.col1.approx_eq_eps(other.col1, epsilon)
    }

    /// Returns `true` if any of the components are NaN
    pub fn is_nan(&self) -> bool {
        self.col0.is_nan() || self.col1.is_nan()
    }

    /// Returns `true` if all of the components are finite
    pub fn is_finite(&self) -> bool {
        self.col0.is_finite() && self.col1.is_finite()
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
        (self.transpose() * *self).approx_eq_eps(Mat2::IDENTITY, epsilon)
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

/// Adds two matrices together component-wise.
impl Add for Mat2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.col0 + rhs.col0, self.col1 + rhs.col1)
    }
}

/// Subtracts `rhs` from `self` component-wise.
impl Sub for Mat2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.col0 - rhs.col0, self.col1 - rhs.col1)
    }
}

/// Multiplies two matrices using standard matrix multiplication.
impl Mul for Mat2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self * rhs.col0, self * rhs.col1)
    }
}

/// Multiplies the matrix by a `Vec2` (matrix-vector multiplication).
impl Mul<Vec2> for Mat2 {
    type Output = Vec2;
    #[inline]
    fn mul(self, rhs: Vec2) -> Self::Output {
        self.col0 * rhs.x + self.col1 * rhs.y
    }
}

/// Multiplies each component of the matrix by a scalar.
impl Mul<f32> for Mat2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.col0 * rhs, self.col1 * rhs)
    }
}

/// Multiplies a scalar by each component of the matrix.
impl Mul<Mat2> for f32 {
    type Output = Mat2;
    #[inline]
    fn mul(self, rhs: Mat2) -> Self::Output {
        rhs * self
    }
}

/// Divides each component of the matrix by a scalar.
impl Div<f32> for Mat2 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.col0 / rhs, self.col1 / rhs)
    }
}

/// Negates each component of the matrix.
impl Neg for Mat2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.col0, -self.col1)
    }
}

// ============= Assignment Operator Overloads =============

impl AddAssign for Mat2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.col0 += rhs.col0;
        self.col1 += rhs.col1;
    }
}

impl SubAssign for Mat2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.col0 -= rhs.col0;
        self.col1 -= rhs.col1;
    }
}

impl MulAssign for Mat2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<f32> for Mat2 {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.col0 *= rhs;
        self.col1 *= rhs;
    }
}

impl DivAssign<f32> for Mat2 {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.col0 /= rhs;
        self.col1 /= rhs;
    }
}

// ============= Trait Implementations =============

impl Default for Mat2 {
    /// Returns the identity matrix.
    #[inline]
    fn default() -> Self {
        Self::IDENTITY // Assumes Mat2::IDENTITY constant exists
    }
}

/// Checks whether two matrices are exactly equal.
impl PartialEq for Mat2 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.col0 == other.col0 && self.col1 == other.col1
    }
}

/// Enables `m[column]` access. Panics if `col_index` is out of bounds.
impl Index<usize> for Mat2 {
    type Output = Vec2;
    #[inline]
    fn index(&self, col_index: usize) -> &Self::Output {
        match col_index {
            0 => &self.col0,
            1 => &self.col1,
            _ => panic!("Mat2 column index out of bounds: {}", col_index),
        }
    }
}

/// Enables mutable `m[column]` access. Panics if `col_index` is out of bounds.
impl IndexMut<usize> for Mat2 {
    #[inline]
    fn index_mut(&mut self, col_index: usize) -> &mut Self::Output {
        match col_index {
            0 => &mut self.col0,
            1 => &mut self.col1,
            _ => panic!("Mat2 column index out of bounds: {}", col_index),
        }
    }
}

/// Implements the `Display` trait for pretty-printing.
impl fmt::Display for Mat2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:.3}, {:.3}]\n[{:.3}, {:.3}]",
            self.col0.x, self.col1.x, self.col0.y, self.col1.y
        )
    }
}

// ============= Approx Crate Implementations =============

/// Implements absolute difference equality comparison for `Mat2`.
impl approx::AbsDiffEq for Mat2 {
    type Epsilon = f32;

    #[inline]
    fn default_epsilon() -> f32 {
        f32::EPSILON
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: f32) -> bool {
        self.col0.abs_diff_eq(&other.col0, epsilon) && self.col1.abs_diff_eq(&other.col1, epsilon)
    }
}

/// Implements relative equality comparison for `Mat2`.
impl approx::RelativeEq for Mat2 {
    #[inline]
    fn default_max_relative() -> f32 {
        f32::EPSILON
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: f32, max_relative: f32) -> bool {
        self.col0.relative_eq(&other.col0, epsilon, max_relative)
            && self.col1.relative_eq(&other.col1, epsilon, max_relative)
    }
}
