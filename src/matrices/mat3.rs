use crate::mat4::Mat4;
use crate::utils::epsilon_eq_default;
use crate::vec::Vec2;
use crate::vec::Vec3;
use crate::vec::Vec4;
use core::f32;
use std::ops::{Add, Div, Mul, Sub};

/// A 3x3 column-major matrix, primarily used for 2D affine transformations.
///
/// An affine transformation combines linear transformations (like rotation, scaling, and shearing)
/// with translation. This matrix type is a cornerstone for 2D graphics and physics, allowing you to
/// represent the position, rotation, and scale of an object in a single entity.
///
/// The matrix is composed of three `Vec3` column vectors. For 2D affine transformations,
/// the layout has a conventional meaning:
///
/// ```text
/// // [col0] [col1] [col2]
/// [ sx  , kx  , tx   ]  // sx: scale x, kx: shear x, tx: translate x
/// [ ky  , sy  , ty   ]  // ky: shear y, sy: scale y, ty: translate y
/// [ 0.0 , 0.0 , 1.0  ]
/// ```
///
/// It can also be used for 3D linear transformations (rotation and scaling),
/// where the third column and row are relevant to the z-axis.
///
/// # Example
/// ```rust
/// use omelet::vec::{Vec3, Vec2};
/// use omelet::matrices::Mat3;
///
/// // A transform that scales by 2x and then moves (5, 10)
/// let scale = Mat3::from_scale(Vec2::new(2.0, 2.0));
/// let translate = Mat3::from_translation(Vec2::new(5.0, 10.0));
/// let transform = translate * scale;
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    /// The first column of the matrix.
    ///
    /// For a 2d affine transform, this represents the **X basis vector**
    /// and controls the scaling and shearing along the x-axis.
    pub col0: Vec3,
    /// The second column of the matrix.
    ///
    /// For 2D affine transform, this represents the **Y basis vector**
    /// and controls the scaling and shearing along the y-axis.
    pub col1: Vec3,
    /// The third column of the matrix.
    ///
    /// For a 2D affine transform, this represents the **translation vector**.
    pub col2: Vec3,
}

impl Mat3 {
    // ============= Construction =============
    /// Constructs a new 3x3 matrix from two column vectors
    ///
    /// The matrix is structured in column-major order:
    /// ```text
    /// [col0.x, col1.x, col2.x]
    /// [col0.y, col1.y, col2.y]
    /// [col0.z, col1.z, col2.z]
    /// ```
    pub fn new(col0: Vec3, col1: Vec3, col2: Vec3) -> Mat3 {
        Mat3 { col0, col1, col2 }
    }

    /// Returns the 3x3 identity matrix
    ///
    /// # Example
    /// ```rust
    /// use omelet::vec::Vec3;
    /// use omelet::matrices::Mat3;
    /// let identity = Mat3::identity();
    /// assert_eq!(identity, Mat3::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)));
    /// ```
    pub fn identity() -> Mat3 {
        Mat3::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
    }

    /// Creates a 3x3 matrix with all elements set to **zero**.
    ///
    /// A zero matrix represents a transformation that collapses any point
    /// or vector to the origin.
    pub fn zero() -> Mat3 {
        Mat3::new(Vec3::zero(), Vec3::zero(), Vec3::zero())
    }

    /// Creates a 3x3 matrix from three **row vectors**.
    ///
    /// This is a convenience method for when you have matrix data
    /// in row-major order. The contructor will rearrange the elements
    /// into the internal column-major format.
    ///
    /// # Example
    /// ```rust
    /// use omelet::vec::Vec3;
    /// use omelet::matrices::Mat3;
    ///
    /// // These rows...
    /// let r0 = Vec3::new(1.0, 2.0, 3.0);
    /// let r1 = Vec3::new(4.0, 5.0, 6.0);
    /// let r2 = Vec3::new(7.0, 8.0, 9.0);
    ///
    /// // ...become these columns
    /// let c0 = Vec3::new(1.0, 4.0, 7.0);
    /// let c1 = Vec3::new(2.0, 5.0, 8.0);
    /// let c2 = Vec3::new(3.0, 6.0, 9.0);
    ///
    /// assert_eq!(Mat3::from_rows(r0, r1, r2), Mat3::new(c0, c1, c2));
    /// ```
    pub fn from_rows(r0: Vec3, r1: Vec3, r2: Vec3) -> Mat3 {
        Mat3::new(
            Vec3::new(r0.x, r1.x, r2.x),
            Vec3::new(r0.y, r1.y, r2.y),
            Vec3::new(r0.z, r1.z, r2.z),
        )
    }

    // ============= Transformation Constructors ==============
    /// Creates a 3x3 rotation matrix from an angle (in radians) and an axis.
    ///
    /// The axis will be normalized internally.
    ///
    /// The matrix is constructed in column-major order:
    /// ```text
    /// [col0.x, col1.x, col2.x]
    /// [col0.y, col1.y, col2.y]
    /// [col0.z, col1.z, col2.z]
    /// ```
    ///
    /// # Parameters
    /// - `radians`: The angle of rotation in radians
    /// - `axis`: The rotation axis `Vec3`. It will be normalized
    ///
    /// # Returns
    /// A `Mat3` representing the rotation
    pub fn from_angle_axis(radians: f32, axis: Vec3) -> Mat3 {
        let n_axis = axis.normalize();
        let x = n_axis.x;
        let y = n_axis.y;
        let z = n_axis.z;

        let cos_theta = radians.cos();
        let sin_theta = radians.sin();
        let one_minus_cos_theta = 1.0 - cos_theta;

        let col0_x = cos_theta + x * x * one_minus_cos_theta;
        let col0_y = y * x * one_minus_cos_theta + z * sin_theta;
        let col0_z = z * x * one_minus_cos_theta - y * sin_theta;
        let final_col0 = Vec3::new(col0_x, col0_y, col0_z);

        let col1_x = x * y * one_minus_cos_theta - z * sin_theta;
        let col1_y = cos_theta + y * y * one_minus_cos_theta;
        let col1_z = z * y * one_minus_cos_theta + x * sin_theta;
        let final_col1 = Vec3::new(col1_x, col1_y, col1_z);

        let col2_x = x * z * one_minus_cos_theta + y * sin_theta;
        let col2_y = y * z * one_minus_cos_theta - x * sin_theta;
        let col2_z = cos_theta + z * z * one_minus_cos_theta;
        let final_col2 = Vec3::new(col2_x, col2_y, col2_z);

        Mat3::new(final_col0, final_col1, final_col2)
    }

    /// Creates a 2D **rotation matrix** from an angle in radians.
    ///
    /// This matrix rotates points and vectors counter-clockwise around the origin (or Z-axis).
    ///
    /// # Parameters
    /// * `radians`: The rotation angle in radians.
    pub fn from_angle_z(radians: f32) -> Mat3 {
        let (s, c) = radians.sin_cos();
        Mat3::new(
            Vec3::new(c, s, 0.0),
            Vec3::new(-s, c, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
    }

    /// Creates a 2D **shearing matrix**.
    ///
    /// A shear transformation slants the shape of an object.
    /// For example, a shear along the x-axis will make a square
    /// into a parallelogram.
    ///
    /// # Parameters
    /// - `shear`: A `Vec2` where `shear.x` is the horizontal shear factor and
    /// `shear.y` is the vertical shear factor.
    pub fn from_shear(shear: Vec2) -> Mat3 {
        Mat3::new(
            Vec3::new(1.0, shear.y, 0.0),
            Vec3::new(shear.x, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
    }

    /// Creates a combined 2D transformation matrix from **Translation, Rotation, and Scale**
    ///
    /// This is a high-level helper for creating a complete object transform.
    /// It correctly applies transformations around a `pivot` point. The order
    /// of operations is: scale, then rotate (both around the pivot), then translate.
    ///
    /// # Parameters
    /// * `translation`: The final position of the object's pivot.
    /// * `angle`: The rotation in radians.
    /// * `scale`: The scale factors.
    /// * `pivot`: The local point to scale and rotate around. For an object rotating
    ///   around its center, this would be its center point in local space.
    pub fn from_trs(translation: Vec2, angle: f32, scale: Vec2, pivot: Vec2) -> Mat3 {
        Mat3::from_translation(translation)
            * Mat3::from_translation(pivot)
            * Mat3::from_angle_z(angle)
            * Mat3::from_scale(scale)
            * Mat3::from_translation(-pivot)
    }

    /// Creates a 2D **translation matrix**.
    ///
    /// This matrix can be used to move (translate) points in 2D space.
    ///
    /// # Parameters
    /// - `translation`: The `Vec2` represenrting the movement along the X and Y axes.
    pub fn from_translation(translation: Vec2) -> Mat3 {
        Mat3::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(translation.x, translation.y, 1.0),
        )
    }

    /// Creates a 2D **scaling matrix**.
    ///
    /// This matrix scales points and vectors away from the origin.
    ///
    /// # Parameters
    /// - `scale`: The `Vec2` representing the scaling factor along the X and Y axes.
    /// A value of `(1.0, 1.0)` results in no scaling.
    pub fn from_scale(scale: Vec2) -> Mat3 {
        Mat3::new(
            Vec3::new(scale.x, 0.0, 0.0),
            Vec3::new(0.0, scale.y, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
    }

    // ============= Core Matrix Operations ==============
    /// Computes the **transpose** of the matrix.
    ///
    /// The transpose of a matrix flips it over its main diagonal, effectively
    /// swapping the row and column indices. For orthogonal matrices (like pure
    /// rotations), the transpose is equal to its inverse, making it a very
    /// cheap way to undo a rotation.
    pub fn transpose(&self) -> Mat3 {
        Mat3::new(
            Vec3::new(self.col0.x, self.col1.x, self.col2.x),
            Vec3::new(self.col0.y, self.col1.y, self.col2.y),
            Vec3::new(self.col0.z, self.col1.z, self.col2.z),
        )
    }

    /// Calculates the **determinant** of the matrix.
    ///
    /// The determinant is a scalar value with important properties.
    /// - If determinant is `0`, the matrix is "singular" and cannot be inverted.
    /// - For 2D affine transforms, its absolute value represents the change in area
    /// caused by the transformation. A determinant of `2.0` means the area doubles.
    /// - A negative determinant indicates that the transformation includes a reflection
    /// (it "flips" the space).
    pub fn determinant(&self) -> f32 {
        self.col0.x * (self.col1.y * self.col2.z - self.col2.y * self.col1.z)
            - self.col1.x * (self.col0.y * self.col2.z - self.col2.y * self.col0.z)
            + self.col2.x * (self.col0.y * self.col1.z - self.col1.y * self.col0.z)
    }

    /// Computes the **inverse** of the matrix, if it exists.
    ///
    /// The inverse is a matrix that "undoes" the transformation of the original.
    /// `M * M.inverse() = Identity`. It is essential for tasks like converting
    /// coordinates from world space back to a model's local space.
    ///
    /// # Returns
    /// `None` if the matrix is not invertible (i.e., its determinant is zero).
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() <= f32::EPSILON {
            return None;
        }
        Some(self.adjugate() / det)
    }

    /// Computes the inverse of the matrix, or returns the identity matrix if
    /// it's not invertible.
    /// A convenient alternatiuve to `inverse()` when you need a valid matrix in all cases.
    pub fn inverse_or_identity(&self) -> Mat3 {
        self.inverse().unwrap_or_else(Mat3::identity)
    }

    /// Returns the specified row vector.
    ///
    /// # Panics
    /// Panics if `row_idx` is not 0, 1 or 2
    pub fn row(&self, row_idx: usize) -> Vec3 {
        match row_idx {
            0 => Vec3::new(self.col0.x, self.col1.x, self.col2.x),
            1 => Vec3::new(self.col0.y, self.col1.y, self.col2.y),
            2 => Vec3::new(self.col0.z, self.col1.z, self.col2.z),
            _ => panic!("Mat3 column index out of bounds: {}", row_idx),
        }
    }

    // ============= Graphics and View Matrices ==============
    /// Creates a 2D **view matrix** that looks from an `eye` position towards a `target`.
    ///
    /// This is essential for camera implementation. It creates a transformation that
    /// places the `eye` at the origin and orients the world so that the `target`
    /// is in view. Applying this matrix to scene objects transforms them from "world space"
    //  to "view space" (or "camera space").
    pub fn look_at(eye: Vec2, target: Vec2) -> Mat3 {
        let fwd = (target - eye).normalize_or_zero();
        // Standard right-handed 2D perpendicular vector
        let right = Vec2::new(-fwd.y, fwd.x);
        Mat3::from_rows(
            Vec3::new(right.x, right.y, -eye.dot(right)),
            Vec3::new(fwd.x, fwd.y, -eye.dot(fwd)),
            Vec3::new(0.0, 0.0, 1.0),
        )
    }

    /// Creates a 2D **orthographic projection matrix**.
    ///
    /// This matrix defines a rectangular viewing volume and map severything
    /// inside it to a standard cube (or square in 2D) known as "normalized device
    /// coordinates" (NDC). It's a projection without any perspective distortion.
    ///
    /// # Parameters
    /// - `left`, `right`: The vertical boundaries of the viewing volume.
    /// - `bottom`, `top`: The horizontal boundaries of the viewing volume.
    pub fn ortho(left: f32, right: f32, bottom: f32, top: f32) -> Mat3 {
        let r_minus_l = right - left;
        let t_minus_b = top - bottom;

        Mat3::new(
            Vec3::new(2.0 / r_minus_l, 0.0, 0.0),
            Vec3::new(0.0, 2.0 / t_minus_b, 0.0),
            Vec3::new(
                -(right + left) / r_minus_l,
                -(top + bottom) / t_minus_b,
                1.0,
            ),
        )
    }

    /// Creates a matrix that transforms points from one rectangle to another.
    ///
    /// This is useful for UI layout, or mapping texture coordinates from
    /// a sprite sheet to a quad.
    ///
    /// # Parameters
    /// - `from`: A tuple `(min, max)` defining the source rectangle.
    /// - `to`: A tuple `(min, max)` defining the destination rectangle.
    pub fn rect_transform(from: (Vec2, Vec2), to: (Vec2, Vec2)) -> Mat3 {
        let from_size = from.1 - from.0;
        let to_size = to.1 - to.0;
        let scale = to_size / from_size;

        Mat3::from_translation(to.0) * Mat3::from_scale(scale) * Mat3::from_translation(-from.0)
    }

    /// Creates a matrix that scales `content` to fit inside a `container` while maintaining
    /// aspect ratio.
    ///
    /// The content will be centered within the container. This is useful for displaying
    /// images or game views within a fixed window or viewport without distortion.
    pub fn fit_rect(container: Vec2, content: Vec2) -> Mat3 {
        let scale_ratio = (container.x / content.x).min(container.y / content.y);
        let scaled_content = content * scale_ratio;
        let offset = (container - scaled_content) * 0.5;

        Mat3::from_translation(offset) * Mat3::from_scale(Vec2::new(scale_ratio, scale_ratio))
    }

    // ============= Math Utilities =============

    /// Returns a matrix with the absolute value of each component
    pub fn abs(self) -> Mat3 {
        Mat3::new(self.col0.abs(), self.col1.abs(), self.col2.abs())
    }

    /// Returns the sign (-1.0, 0.0, 1.0) of each matrix component
    pub fn signum(self) -> Mat3 {
        Mat3::new(self.col0.signum(), self.col1.signum(), self.col2.signum())
    }

    /// Computes the **adjugate** (or classical adjoint) of the matrix.
    ///
    /// The adjugate is the transpose of the cofactor matrix. It's mainly used
    /// as an intermediate step for calculating the matrix inverse.
    /// `inverse(M) = adjugate(M) / determinant(M)`.
    pub fn adjugate(&self) -> Mat3 {
        let m = self;
        Mat3::new(
            Vec3::new(
                m.col1.y * m.col2.z - m.col1.z * m.col2.y, // Cofactor of m.col0.x (m00)
                m.col0.z * m.col2.y - m.col0.y * m.col2.z, // Cofactor of m.col1.x (m01) -- this is a bit off based on typical adjugate derivation
                m.col0.y * m.col1.z - m.col0.z * m.col1.y, // Cofactor of m.col2.x (m02) -- this is a bit off
            ),
            Vec3::new(
                m.col1.z * m.col2.x - m.col1.x * m.col2.z, // Cofactor of m.col0.y (m10) -- this is a bit off
                m.col0.x * m.col2.z - m.col0.z * m.col2.x, // Cofactor of m.col1.y (m11)
                m.col0.z * m.col1.x - m.col0.x * m.col1.z, // Cofactor of m.col2.y (m12) -- this is a bit off
            ),
            Vec3::new(
                m.col1.x * m.col2.y - m.col1.y * m.col2.x, // Cofactor of m.col0.z (m20) -- this is a bit off
                m.col0.y * m.col2.x - m.col0.x * m.col2.y, // Cofactor of m.col1.z (m21) -- this is a bit off
                m.col0.x * m.col1.y - m.col0.y * m.col1.x, // Cofactor of m.col2.z (m22)
            ),
        )
    }

    // ============= Decomposition / Extraction ==============
    /// Extracts the **scale** component from the matrix by computing
    /// the length (magnitude) of each column vector
    ///
    /// This assumes that the matrix encodes a rotation + scale transform
    pub fn get_scale(&self) -> Vec2 {
        Vec2::new(
            Vec2::new(self.col0.x, self.col0.y).length(),
            Vec2::new(self.col1.x, self.col1.y).length(),
        )
    }

    /// Extracts the **rotation** angle in radians from the matrix
    ///
    /// Assumes the matrix is a pure rotation or a rotation + scale transform
    /// Uses the direction of the first column vector (x-axis) to determine angle
    pub fn get_rotation(&self) -> f32 {
        let norm_x = Vec2::new(self.col0.x, self.col0.y).normalize();
        norm_x.y.atan2(norm_x.x)
    }

    /// Extracts the **translation** component from an affine transformation matrix.
    /// This is simply the `(x, y)` part of the third column.
    pub fn get_translation(&self) -> Vec2 {
        Vec2::new(self.col2.x, self.col2.y)
    }

    /// Decomposes the matrix into its `(translation, rotation, scale)` components.
    ///
    /// This is useful for inspecting the properties of a transformation matrix,
    /// for example, to display them in an editor or to contrain one of the components.
    /// # Returns
    /// A tuple of `(translation, rotation_radians, scale)`.
    pub fn decompose(&self) -> (Vec2, f32, Vec2) {
        (
            self.get_translation(),
            self.get_rotation(),
            self.get_scale(),
        )
    }

    // ============= Transformations ==============
    /// Transforms a 2D **point** by the matrix.
    ///
    /// Points are affected by the full affine transformation, including translation.
    /// This is mathematically equivalent to multiplying the matrix by a
    /// `Vec3(point.x, point.y, 1.0)`.
    pub fn transform_point(&self, point: Vec2) -> Vec2 {
        let v = *self * (Vec3::new(point.x, point.y, 1.0));
        Vec2::new(v.x, v.y)
    }

    /// Transforms a 2D **vector** by the matrix.
    ///
    /// Vectors represent directions and magnitudes, so they are affected by rotation
    /// and scaling, but **not** by translation. This is mathematically equivalent
    /// to multiplying the matrix by a `Vec3(vector.x, vector.y, 0.0)`.
    pub fn transform_vector(&self, vector: Vec2) -> Vec2 {
        let v = *self * (Vec3::new(vector.x, vector.y, 0.0));
        Vec2::new(v.x, v.y)
    }

    /// Transforms an axis-aligned bounding box (AABB) and returns the new AABB
    /// that encloses it.
    ///
    /// Since a rotated rectangle is no longer axis-aligned, this function calculates
    /// the minimum and maximum corners of a new AABB that can fully contain the
    /// transformed original.
    pub fn transform_aabb(&self, min: Vec2, max: Vec2) -> (Vec2, Vec2) {
        let corners = [
            self.transform_point(min),
            self.transform_point(Vec2::new(min.x, max.y)),
            self.transform_point(Vec2::new(max.x, min.y)),
            self.transform_point(max),
        ];

        let mut new_min = corners[0];
        let mut new_max = corners[0];

        for i in 1..4 {
            new_min = new_min.min(corners[i]);
            new_max = new_max.max(corners[i]);
        }

        (new_min, new_max)
    }

    // ============= Builder Methods ==============
    /// Applies a translation to the matrix (non-consuming)
    pub fn apply_translation(&mut self, translation: Vec2) {
        *self = Mat3::from_translation(translation) * *self
    }

    /// Applies a rotation to the matrix (non-consuming)
    pub fn apply_rotation(&mut self, angle_rad: f32) {
        *self = Mat3::from_angle_z(angle_rad) * *self
    }

    /// Returns a new matrix with a translation applied (consuming)
    pub fn with_translation(self, translation: Vec2) -> Mat3 {
        Mat3::from_translation(translation) * self
    }

    /// Returns a new matrix with a rotation applied (consuming)
    pub fn with_rotation(self, angle_rad: f32) -> Mat3 {
        Mat3::from_angle_z(angle_rad) * self
    }

    // ============= Utilities & Checks ==============
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
    pub fn lerp(&self, b: Mat3, t: f32) -> Mat3 {
        Mat3::new(
            self.col0.lerp(b.col0, t),
            self.col1.lerp(b.col1, t),
            self.col2.lerp(b.col2, t),
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
    pub fn lerp_between(a: Mat3, b: Mat3, t: f32) -> Mat3 {
        Mat3::new(
            a.col0.lerp(b.col0, t),
            a.col1.lerp(b.col1, t),
            a.col2.lerp(b.col2, t),
        )
    }

    /// Returns the **diagonal** vector of the matrix.
    pub fn get_diagonal(&self) -> Vec2 {
        Vec2::new(self.col0.x, self.col1.y)
    }

    /// Returns the **trace** of the matrix.
    ///
    /// The trace is the sum of the elements on the main diagonal.
    /// While less common in graphics, it has uses in linear algebra
    /// and physics.
    pub fn trace(&self) -> f32 {
        self.col0.x + self.col1.y + self.col2.z
    }

    /// Checks if the matrix is **invertible** (i.e., its determinant is non-zero).
    /// A non-invertible matrix represents a transformation that collapses space
    /// and cannot be undone.
    pub fn is_invertible(&self) -> bool {
        let det = self.determinant();
        det.abs() > f32::EPSILON && det.is_finite()
    }

    /// Checks if the matrix is the **identity** matrix.
    pub fn is_identity(&self) -> bool {
        *self == Mat3::identity()
    }

    /// Checks if the matrix represents a valid 2D **affine** transformation.
    /// This is true if its bottom row is `(0, 0, 1)`, which ensures that the
    /// `w` component of the transformed points remains `1`.
    pub fn is_affine(&self) -> bool {
        epsilon_eq_default(self.col0.z, 0.0)
            && epsilon_eq_default(self.col1.z, 0.0)
            && epsilon_eq_default(self.col2.z, 1.0)
    }

    /// Checks if the matrix contains **mirroring** or flipping.
    /// This is true if the determinant is negative, which means the orientation
    /// of the space has been inverted (e.g., from right-handed to left-handed).
    pub fn has_mirroring(&self) -> bool {
        self.determinant() < 0.0
    }

    /// Checks if matrices are approx equal using default epsilon
    pub fn approx_eq(&self, other: Mat3) -> bool {
        self.col0.approx_eq(other.col0)
            && self.col1.approx_eq(other.col1)
            && self.col2.approx_eq(other.col2)
    }

    /// Checks if matrices are approx equal using user entered epsilon
    pub fn approx_eq_eps(&self, other: Mat3, epsilon: f32) -> bool {
        self.col0.approx_eq_eps(other.col0, epsilon)
            && self.col1.approx_eq_eps(other.col1, epsilon)
            && self.col2.approx_eq_eps(other.col2, epsilon)
    }

    /// Returns `true` if any of the components are NaN
    pub fn is_nan(&self) -> bool {
        return self.col0.is_nan() || self.col1.is_nan() || self.col2.is_nan();
    }

    /// Returns `true` if all of the components are finite
    pub fn is_finite(&self) -> bool {
        return self.col0.is_finite() && self.col1.is_finite() && self.col2.is_finite();
    }

    /// Converts this 2D affine `Mat3` to a `Mat4`.
    ///
    /// This is the standard way to promote a 2D transform into a 3D space for
    /// use in a 3D rendering pipeline. It correctly places the 2D components
    /// into the 4x4 matrix, leaving the Z-axis transformations as identity.
    pub fn to_mat4_affine(&self) -> Mat4 {
        Mat4::new(
            Vec4::new(self.col0.x, self.col0.y, 0.0, 0.0),
            Vec4::new(self.col1.x, self.col1.y, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(self.col2.x, self.col2.y, 0.0, 1.0),
        )
    }

    /// Returns a row-major 2D array `[[f32; 3]; 3]`
    ///
    /// The matrix is composed of column vectors, so to get rows,
    /// we transpose it mentally or explicitly.
    ///
    /// Equivalent to:
    /// ```text
    /// [[col0.x, col1.x, col2.x],
    ///  [col0.y, col1.y, col2.y],
    ///  [col0.z, col1.z, col2.z]]
    /// ```
    pub fn to_array_2d_row_major(&self) -> [[f32; 3]; 3] {
        [
            [self.col0.x, self.col1.x, self.col2.x], // Row 0
            [self.col0.y, self.col1.y, self.col2.y], // Row 1
            [self.col0.z, self.col1.z, self.col2.z], // Row 2
        ]
    }

    /// Returns a row-major flat array `[f32; 9]`
    ///
    /// Equivalent to `[col0.x, col1.x, col2.x, col0.y, col1.y, col2.y, col0.z, col1.z, col2.z]`
    pub fn to_array_row_major(&self) -> [f32; 9] {
        [
            self.col0.x,
            self.col1.x,
            self.col2.x, // Row 0
            self.col0.y,
            self.col1.y,
            self.col2.y, // Row 1
            self.col0.z,
            self.col1.z,
            self.col2.z, // Row 2
        ]
    }

    /// Returns a column-major 2D array `[[f32; 3]; 3]`
    ///
    /// This directly corresponds to how the `Mat3` is stored (columns as Vec3s).
    ///
    /// Equivalent to:
    /// ```text
    /// [[col0.x, col0.y, col0.z],
    ///  [col1.x, col1.y, col1.z],
    ///  [col2.x, col2.y, col2.z]]
    /// ```
    pub fn to_array_2d_col_major(&self) -> [[f32; 3]; 3] {
        [
            [self.col0.x, self.col0.y, self.col0.z], // col0 components
            [self.col1.x, self.col1.y, self.col1.z], // col1 components
            [self.col2.x, self.col2.y, self.col2.z], // col2 components
        ]
    }

    /// Returns a column-major flat array `[f32; 9]`
    ///
    /// Equivalent to `[col0.x, col0.y, col0.z, col1.x, col1.y, col1.z, col2.x, col2.y, col2.z]`
    pub fn to_array_col_major(&self) -> [f32; 9] {
        [
            self.col0.x,
            self.col0.y,
            self.col0.z, // col0 components
            self.col1.x,
            self.col1.y,
            self.col1.z, // col1 components
            self.col2.x,
            self.col2.y,
            self.col2.z, // col2 components
        ]
    }

    /// Returns a row-major 2D tuple `((f32, f32, f32), (f32, f32, f32), (f32, f32, f32))`
    ///
    /// Equivalent to `((col0.x, col1.x, col2.x), (col0.y, col1.y, col2.y), (col0.z, col1.z, col2.z))`
    pub fn to_tuple_2d_row_major(&self) -> ((f32, f32, f32), (f32, f32, f32), (f32, f32, f32)) {
        (
            (self.col0.x, self.col1.x, self.col2.x), // Row 0
            (self.col0.y, self.col1.y, self.col2.y), // Row 1
            (self.col0.z, self.col1.z, self.col2.z), // Row 2
        )
    }

    /// Returns a row-major flat tuple `(f32, f32, f32, f32, f32, f32, f32, f32, f32)`
    ///
    /// Equivalent to `(col0.x, col1.x, col2.x, col0.y, col1.y, col2.y, col0.z, col1.z, col2.z)`
    pub fn to_tuple_row_major(&self) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32) {
        (
            self.col0.x,
            self.col1.x,
            self.col2.x, // Row 0
            self.col0.y,
            self.col1.y,
            self.col2.y, // Row 1
            self.col0.z,
            self.col1.z,
            self.col2.z, // Row 2
        )
    }

    /// Returns a column-major 2D tuple `((f32, f32, f32), (f32, f32, f32), (f32, f32, f32))`
    ///
    /// Equivalent to `((col0.x, col0.y, col0.z), (col1.x, col1.y, col1.z), (col2.x, col2.y, col2.z))`
    pub fn to_tuple_2d_col_major(&self) -> ((f32, f32, f32), (f32, f32, f32), (f32, f32, f32)) {
        (
            (self.col0.x, self.col0.y, self.col0.z), // col0 components
            (self.col1.x, self.col1.y, self.col1.z), // col1 components
            (self.col2.x, self.col2.y, self.col2.z), // col2 components
        )
    }

    /// Returns a column-major flat tuple `(f32, f32, f32, f32, f32, f32, f32, f32, f32)`
    ///
    /// Equivalent to `(col0.x, col0.y, col0.z, col1.x, col1.y, col1.z, col2.x, col2.y, col2.z)`
    pub fn to_tuple_col_major(&self) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32) {
        (
            self.col0.x,
            self.col0.y,
            self.col0.z, // col0 components
            self.col1.x,
            self.col1.y,
            self.col1.z, // col1 components
            self.col2.x,
            self.col2.y,
            self.col2.z, // col2 components
        )
    }

    /// Returns a `Mat3` from a 2D array
    ///
    /// # Parameters
    /// - `arr`: 2D array `[[f32; 3]; 3]` representing a **row-major** matrix.
    ///
    /// Note: Since Mat3 stores columns, we need to convert the row-major input
    /// into column vectors for the Mat3 constructor.
    pub fn from_2d_array(arr: [[f32; 3]; 3]) -> Mat3 {
        // arr[0] is row 0: [m00, m01, m02]
        // arr[1] is row 1: [m10, m11, m12]
        // arr[2] is row 2: [m20, m21, m22]
        Mat3::new(
            Vec3::new(arr[0][0], arr[1][0], arr[2][0]), // col0: (m00, m10, m20)
            Vec3::new(arr[0][1], arr[1][1], arr[2][1]), // col1: (m01, m11, m21)
            Vec3::new(arr[0][2], arr[1][2], arr[2][2]), // col2: (m02, m12, m22)
        )
    }

    /// Returns a `Mat3` from a flat array
    ///
    /// # Parameters
    /// - `arr`: Flat array `[f32; 9]` representing a **row-major** matrix.
    ///
    /// Equivalent to `[m00, m01, m02, m10, m11, m12, m20, m21, m22]`
    pub fn from_array(arr: [f32; 9]) -> Mat3 {
        Mat3::new(
            Vec3::new(arr[0], arr[3], arr[6]), // col0: (m00, m10, m20)
            Vec3::new(arr[1], arr[4], arr[7]), // col1: (m01, m11, m21)
            Vec3::new(arr[2], arr[5], arr[8]), // col2: (m02, m12, m22)
        )
    }

    /// Returns a `Mat3` from a 2D tuple
    ///
    /// # Parameters
    /// - `t`: Tuple `((f32, f32, f32), (f32, f32, f32), (f32, f32, f32))` representing a **row-major** matrix.
    pub fn from_2d_tuple(t: ((f32, f32, f32), (f32, f32, f32), (f32, f32, f32))) -> Mat3 {
        // t.0 is row 0: (m00, m01, m02)
        // t.1 is row 1: (m10, m11, m12)
        // t.2 is row 2: (m20, m21, m22)
        Mat3::new(
            Vec3::new(t.0.0, t.1.0, t.2.0), // col0: (m00, m10, m20)
            Vec3::new(t.0.1, t.1.1, t.2.1), // col1: (m01, m11, m21)
            Vec3::new(t.0.2, t.1.2, t.2.2), // col2: (m02, m12, m22)
        )
    }

    /// Returns a `Mat3` from a flat tuple
    ///
    /// # Parameters
    /// - `t`: Tuple `(f32, f32, f32, f32, f32, f32, f32, f32, f32)` representing a **row-major** matrix.
    ///
    /// Equivalent to `(m00, m01, m02, m10, m11, m12, m20, m21, m22)`
    pub fn from_tuple(t: (f32, f32, f32, f32, f32, f32, f32, f32, f32)) -> Mat3 {
        Mat3::new(
            Vec3::new(t.0, t.3, t.6), // col0: (m00, m10, m20)
            Vec3::new(t.1, t.4, t.7), // col1: (m01, m11, m21)
            Vec3::new(t.2, t.5, t.8), // col2: (m02, m12, m22)
        )
    }
}

//Operator overloads
/// Adds two matrices element-wise.
impl Add for Mat3 {
    type Output = Self;
    fn add(self, rhs: Mat3) -> Mat3 {
        Mat3::new(
            self.col0 + rhs.col0,
            self.col1 + rhs.col1,
            self.col2 + rhs.col2,
        )
    }
}

/// Subtracts the right-hand matrix from the left-hand matrix, element-wise.
impl Sub for Mat3 {
    type Output = Self;
    fn sub(self, rhs: Mat3) -> Mat3 {
        Mat3::new(
            self.col0 - rhs.col0,
            self.col1 - rhs.col1,
            self.col2 - rhs.col2,
        )
    }
}

/// Performs matrix multiplication.
///
/// This is the primary way to combine transformations. Note that matrix
/// multiplication is **not** commutative (`A * B != B * A`). Transformations
/// are applied from right to left. For `T* R * S * point`, the point is
/// first scaled, then rotated, then translated.
impl Mul for Mat3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(self * (rhs.col0), self * (rhs.col1), self * (rhs.col2))
    }
}

/// Multiplies every elemnt of the matrix by a scalar.
impl Mul<f32> for Mat3 {
    type Output = Self;
    fn mul(self, scalar: f32) -> Self {
        Self::new(self.col0 * scalar, self.col1 * scalar, self.col2 * scalar)
    }
}

/// Multiplies a matrix by a scalar from the left-hand side.
impl Mul<Mat3> for f32 {
    type Output = Mat3;
    fn mul(self, mat: Mat3) -> Mat3 {
        mat * self
    }
}

/// Divides every element of the matrix by a scalar.
impl Div<f32> for Mat3 {
    type Output = Self;
    fn div(self, scalar: f32) -> Self {
        Mat3::new(self.col0 / scalar, self.col1 / scalar, self.col2 / scalar)
    }
}

/// Multiplies the matrix with a `Vec3` (matrix-vector multiplication).
impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.col0.x * v.x + self.col1.x * v.y + self.col2.x * v.z,
            self.col0.y * v.x + self.col1.y * v.y + self.col2.y * v.z,
            self.col0.z * v.x + self.col1.z * v.y + self.col2.z * v.z,
        )
    }
}

/// Returns the identity matrix by default.
impl Default for Mat3 {
    fn default() -> Self {
        Mat3::identity()
    }
}

use std::ops::{Index, IndexMut};

/// Enables column access using bracket notation (e.g., `matrix[0]`).
///
/// # Panics
/// Panics if `col_idx` is out of bounds (not 0, 1, 2).
impl Index<usize> for Mat3 {
    type Output = Vec3;

    fn index(&self, col_idx: usize) -> &Vec3 {
        match col_idx {
            0 => &self.col0,
            1 => &self.col1,
            2 => &self.col2,
            _ => panic!("Mat3 row index out of bounds: {}", col_idx),
        }
    }
}

/// Enables mutable column access using bracket notation (e.g., `matrix[0]`).
///
/// # Panics
/// Panics if `col_idx` is out of bounds (not 0, 1, 2).
impl IndexMut<usize> for Mat3 {
    fn index_mut(&mut self, col_idx: usize) -> &mut Vec3 {
        match col_idx {
            0 => &mut self.col0,
            1 => &mut self.col1,
            2 => &mut self.col2,
            _ => panic!("Mat3 row index out of bounds: {}", col_idx),
        }
    }
}
