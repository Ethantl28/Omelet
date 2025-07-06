use crate::mat3::Mat3;
use crate::utils::{epsilon_eq, epsilon_eq_default};
use crate::vec::{Vec2, Vec3};
use core::f32;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

/// 2x2 column-major matrix for linear transformation
///
/// The matrix is represented by two column vectors (`x`, `y`), each a `Vec2`.
/// This layout is suitable for 2D transformations like rotation, scaling, and shearing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat2 {
    pub x: Vec2,
    pub y: Vec2,
}

impl Mat2 {
    ///Creates a new matrix from column vectors.
    ///
    /// # Parameters
    /// - `x`: first column vector.
    /// - `y`: second column vector.
    ///
    /// # Returns
    /// A new `Mat2` with columns `x` and `y`
    pub fn new(x: Vec2, y: Vec2) -> Mat2 {
        Mat2 { x, y }
    }

    /// Creates an identity matrix.
    ///
    /// This is a matrix with `1.0` on the diagonal and `0.0` elsewhere,
    /// which represents the no-op linear transformation.
    pub fn identity() -> Mat2 {
        Mat2::new(Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0))
    }

    /// Creates a zero matrix
    ///
    /// All elements will be zero. This matrix transforms all vectors to zero.
    pub fn zero() -> Mat2 {
        Mat2::new(Vec2::zero(), Vec2::zero())
    }

    /// Creates a diagonal matrix from a vector of diagonal elements
    ///
    /// The off-diagonal elements will be zero.
    ///
    /// # Parameters
    /// - `diag`: Vector containing the diagonal elements `(a, d)`
    ///
    /// # Returns
    /// A matrix:
    /// ```text
    /// [a, 0]
    /// [0, d]
    /// ```
    pub fn from_diagonal(diag: Vec2) -> Mat2 {
        Mat2::new(Vec2::new(diag.x, 0.0), Vec2::new(0.0, diag.y))
    }

    /// Creates a 2D rotation matrix.
    ///
    /// The rotation angle is given in radians. The resulting matrix
    /// rotates vectors counter-clockwise by `angle_rad`.
    ///
    /// Column-major format:
    /// ```text
    /// [ cosθ, sinθ]
    /// [-sinθ, cosθ]
    /// ```
    pub fn from_rotation(angle_rad: f32) -> Mat2 {
        let (sin, cos) = angle_rad.sin_cos();
        Mat2::new(Vec2::new(cos, sin), Vec2::new(-sin, cos))
    }

    /// Creates a scaling matrix.
    ///
    /// Scaling factors for x and y axes are taken from the `scale` vector.
    ///
    /// # Parameters
    /// - `scale`: Scaling factors along the x and y axes.
    ///
    /// # Returns
    /// A diagonal scaling matrix
    pub fn from_scale(scale: Vec2) -> Mat2 {
        Mat2::from_diagonal(scale)
    }

    /// Transposes the matrix.
    ///
    /// Rows become columns and vice versa.
    ///
    /// # Returns
    /// The transposed matrix.
    pub fn transpose(&self) -> Mat2 {
        Mat2::new(Vec2::new(self.x.x, self.y.x), Vec2::new(self.x.y, self.y.y))
    }

    /// Calculates the determinant.
    ///
    /// The determinant indicates the area scaling factor and whether
    /// the transformation preserves orientation.
    ///
    /// # Returns
    /// The scalar determinant value.
    pub fn determinant(&self) -> f32 {
        self.x.x * self.y.y - self.x.y * self.y.x
    }

    /// Computes the inverse of the matrix if it exists..
    ///
    /// Returns `None` if the matrix is singular (determinant near zero)
    /// or if the determinant is not finite.
    ///
    /// # Returns
    /// `Some(inverse_matrix)` if invertible, otherwise `None`.
    pub fn inverse(&self) -> Option<Mat2> {
        let det = self.determinant();
        if det.abs() <= f32::EPSILON || !det.is_finite() {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Mat2::new(
            Vec2::new(self.y.y, -self.x.y) * inv_det,
            Vec2::new(-self.y.x, self.x.x) * inv_det,
        ))
    }

    /// Checks if two matrices are approx equal using the default epsilon.
    ///
    /// This compares each corresponding element of the matrices.
    ///
    /// # Returns
    /// `true` if all elements are approx equal within the default tolerance.
    pub fn approx_eq(&self, other: Mat2) -> bool {
        epsilon_eq_default(self.x.x, other.x.x)
            && epsilon_eq_default(self.x.y, other.x.y)
            && epsilon_eq_default(self.y.x, other.y.x)
            && epsilon_eq_default(self.y.y, other.y.y)
    }

    /// Checks if matrices are approx equal using a user-specified epsilon.
    ///
    /// # Parameters
    /// - `other`: The matrix to compare against.
    /// - `epsilon`: The maximum allowed difference per element.
    ///
    /// # Returns
    /// `true` if all elements differ by less than `epsilon`.
    pub fn approx_eq_eps(&self, other: Mat2, epsilon: f32) -> bool {
        fn epsilon_eq(a: f32, b: f32, epsilon: f32) -> bool {
            (a - b).abs() < epsilon
        }
        epsilon_eq(self.x.x, other.x.x, epsilon)
            && epsilon_eq(self.x.y, other.x.y, epsilon)
            && epsilon_eq(self.y.x, other.y.x, epsilon)
            && epsilon_eq(self.y.y, other.y.y, epsilon)
    }

    /// Returns `true` if all elements are finite number (no NaN or infinity)
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    ///Returns `true` if any element of the matrix is NaN (not a number).
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan()
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
            Vec2::new(self.y.y, -self.x.y),
            Vec2::new(-self.y.x, self.x.x),
        )
    }

    /// Returns the trace of the matrix.
    ///
    /// The trace is the sum of the diagonal elements (`a + d`).
    ///
    /// # Returns
    /// Scalar trace value.
    pub fn trace(self) -> f32 {
        self.x.x + self.y.y
    }

    /// Returns the diagonal elements of the matrix as a vector.
    ///
    /// This represents the scale components if the matrix is diagonal or close.
    ///
    /// # Returns
    /// A vector `[a, d]` where `a` and `d` are diagonal elements.
    pub fn diagonal_scale(self) -> Vec2 {
        Vec2::new(self.x.x, self.y.y)
    }

    /// Returns the scale factors of the matrix.
    ///
    /// This computes the length (magnitude) of each column vector,
    /// representing scaling in each basis direction.
    ///
    /// # Returns
    /// A vector of scale factors.
    pub fn get_scale(&self) -> Vec2 {
        Vec2::new(self.x.length(), self.y.length())
    }

    /// Returns the rotation angle (in radians) encoded by the matrix.
    ///
    /// Returns 0 if the scale is effectively zero (to avoid division by zero).
    ///
    /// # Returns
    /// The rotation angle in radians.
    pub fn get_rotation(&self) -> f32 {
        //Check if scale is 0, if it is, rotation cant be found
        let scale = self.get_scale();
        if scale.x < f32::EPSILON || scale.y < f32::EPSILON {
            return 0.0;
        }

        //Normalize the first basis vector to get pure rotation
        let normalized_x = self.x.normalize();
        normalized_x.y.atan2(normalized_x.x)
    }

    /// Decomposes the matrix into scale and rotation.
    ///
    /// # Returns
    /// Tuple `(scale_vector, rotation_angle_in_radians)`.
    pub fn decompose(&self) -> (Vec2, f32) {
        (self.get_scale(), self.get_rotation())
    }

    /// Returns the specified column vector.
    ///
    /// # Panics
    /// Panics if `col_idx` is not 0 or 1.
    pub fn col(&self, col_idx: usize) -> Vec2 {
        match col_idx {
            0 => Vec2::new(self.x.x, self.y.x),
            1 => Vec2::new(self.x.y, self.y.y),
            _ => panic!("Mat2 column index out of bounds: {}", col_idx),
        }
    }

    /// Creates a matrix from two column vectors
    pub fn from_cols(x: Vec2, y: Vec2) -> Self {
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
        self.x.y.abs() <= epsilon
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
        self.y.x.abs() <= epsilon
    }

    /// Returns the matrix as a 2D row-major array.
    ///
    /// # Returns
    /// Array of rows, each containing 2 elements.
    pub fn to_array_2d_row_major(&self) -> [[f32; 2]; 2] {
        [
            [self.x.x, self.y.x], //Row 0
            [self.x.y, self.y.y], //Row 1
        ]
    }

    /// Returns the matrix as a flat row-major array.
    ///
    /// The order is `[m00, m01, m10, m11]`.
    pub fn to_array_row_major(&self) -> [f32; 4] {
        [self.x.x, self.x.y, self.y.x, self.y.y]
    }

    /// Returns the matrix as a 2D column-major array.
    ///
    /// # Returns
    /// Array of columns, each containing 2 elements.
    pub fn to_array_2d_col_major(&self) -> [[f32; 2]; 2] {
        [[self.x.x, self.x.y], [self.y.x, self.y.y]]
    }

    /// Returns the matrix as a flat column-major array.
    ///
    /// The order is `[m00, m10, m01, m11]`.
    pub fn to_array_col_major(&self) -> [f32; 4] {
        [self.x.x, self.x.y, self.y.x, self.y.y]
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
            Vec3::new(self.x.x, self.x.y, 0.0),
            Vec3::new(self.y.x, self.y.y, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
    }
}

//Operator overloads

/// Adds two matrices element-wise.
///
/// # Returns
/// A matrix where each element is the sum of the corresponding elements in `self` and `rhs`.
///
/// Equivalent to `self + rhs`.
impl Add for Mat2 {
    type Output = Self;
    fn add(self, rhs: Mat2) -> Mat2 {
        Mat2::new(self.x + rhs.x, self.y + rhs.y)
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
        Mat2::new(self.x - rhs.x, self.y - rhs.y)
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

/// Multiplies every element of the matrix by a scalar.
///
/// # Returns
/// A matrix scaled by `scalar`.
///
/// Equivalent to `self * scalar`.
impl Mul<f32> for Mat2 {
    type Output = Self;
    fn mul(self, scalar: f32) -> Mat2 {
        Mat2::new(self.x * scalar, self.y * scalar)
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
        Mat2::new(self.x / scalar, self.y / scalar)
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
            self.x.x * v.x + self.y.x * v.y,
            self.x.y * v.x + self.y.y * v.y,
        )
    }
}

impl approx::AbsDiffEq for Mat2 {
    type Epsilon = f32;

    fn default_epsilon() -> f32 {
        1e-6
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: f32) -> bool {
        epsilon_eq(self.x.x, other.x.x, epsilon)
            && epsilon_eq(self.x.y, other.x.y, epsilon)
            && epsilon_eq(self.y.x, other.y.x, epsilon)
            && epsilon_eq(self.y.y, other.y.y, epsilon)
    }
}

impl approx::RelativeEq for Mat2 {
    fn default_max_relative() -> f32 {
        1e-6
    }

    fn relative_eq(&self, other: &Self, epsilon: f32, max_relative: f32) -> bool {
        // You can reuse approx crate Vec2 relative_eq if available
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

/// Enables `m[row]` access
///
/// # Panics
/// Panics if `row >= 2`.
impl Index<usize> for Mat2 {
    type Output = Vec2;
    fn index(&self, row: usize) -> &Vec2 {
        match row {
            0 => &self.x,
            1 => &self.y,
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
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Mat2 row index out of bounds: {}", row),
        }
    }
}

impl fmt::Display for Mat2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print as 2 rows for readability
        writeln!(f, "[[{:.4}, {:.4}],", self.x.x, self.y.x)?;
        writeln!(f, "[{:.4}, {:.4}]]", self.x.y, self.y.y)
    }
}
