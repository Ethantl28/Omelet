use crate::quaternion::Quat;
use crate::vec::Vec3;
use crate::vec::Vec4;
use core::f32;
use std::{
    cmp::PartialEq,
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// A 4x4 column-major matrix for 3D transformations.
///
/// This matrix type is the foundation for 3D graphics, used to represent transformations
/// like translation, rotation, scaling, and perspective projection. It is stored in
/// **column-major** order, which is the standard layout for graphics APIs like OpenGL and Vulkan.
///
/// The matrix is composed of four `Vec4` column vectors:
///
/// ```text
/// // [col0] [col1] [col2] [col3]
/// [ x.x  ,  y.x ,  z.x ,  w.x ]
/// [ x.y  ,  y.y ,  z.y ,  w.y ]
/// [ x.z  ,  y.z ,  z.z ,  w.z ]
/// [ x.w  ,  y.w ,  z.w ,  w.w ]
/// ```
///
/// For an affine transformation matrix, the columns represent the basis vectors and translation:
/// - `col0`: The X basis vector.
/// - `col1`: The Y basis vector.
/// - `col2`: The Z basis vector.
/// - `col3`: The translation vector.
#[derive(Debug, Clone, Copy)]
pub struct Mat4 {
    pub col0: Vec4,
    pub col1: Vec4,
    pub col2: Vec4,
    pub col3: Vec4,
}

// ============= Types ==============
pub type Mat4Tuple2D = (
    (f32, f32, f32, f32),
    (f32, f32, f32, f32),
    (f32, f32, f32, f32),
    (f32, f32, f32, f32),
);
pub type Mat4Tuple = (
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
);

impl Mat4 {
    // ============= Construction and Conversion =============
    /// Creates a new 4x4 matrix from four column vectors.
    #[inline]
    #[must_use]
    pub fn new(col0: Vec4, col1: Vec4, col2: Vec4, col3: Vec4) -> Mat4 {
        Mat4 {
            col0,
            col1,
            col2,
            col3,
        }
    }

    /// Creates a 4x4 matrix from four **row** vectors.
    ///
    /// This is a convenience method for creating a matrix from data in row-major order.
    /// The constructor will transpose the input to match the internal column-major layout.
    #[inline]
    #[must_use]
    pub fn from_rows(r0: Vec4, r1: Vec4, r2: Vec4, r3: Vec4) -> Mat4 {
        Mat4::new(
            Vec4::new(r0.x, r1.x, r2.x, r3.x),
            Vec4::new(r0.y, r1.y, r2.y, r3.y),
            Vec4::new(r0.z, r1.z, r2.z, r3.z),
            Vec4::new(r0.w, r1.w, r2.w, r3.w),
        )
    }

    /// Creates a `Mat4` from a 16-element array in **column-major** order.
    #[inline]
    #[must_use]
    pub fn from_array(arr: [f32; 16]) -> Mat4 {
        Mat4::new(
            Vec4::new(arr[0], arr[1], arr[2], arr[3]),
            Vec4::new(arr[4], arr[5], arr[6], arr[7]),
            Vec4::new(arr[8], arr[9], arr[10], arr[11]),
            Vec4::new(arr[12], arr[13], arr[14], arr[15]),
        )
    }

    /// Creates a `Mat4` from a 2 dimensional `[[f32; 4]; 4]` array in **column-major** order.
    #[inline]
    #[must_use]
    pub fn from_2d_array(arr: [[f32; 4]; 4]) -> Mat4 {
        Mat4::new(
            Vec4::new(arr[0][0], arr[1][0], arr[2][0], arr[3][0]),
            Vec4::new(arr[0][1], arr[1][1], arr[2][1], arr[3][1]),
            Vec4::new(arr[0][2], arr[1][2], arr[2][2], arr[3][2]),
            Vec4::new(arr[0][3], arr[1][3], arr[2][3], arr[3][3]),
        )
    }

    /// Creates a `Mat4` from a 16-element tuple in **column-major** order.
    #[inline]
    #[must_use]
    pub fn from_tuple(t: Mat4Tuple) -> Mat4 {
        Mat4::new(
            Vec4::new(t.0, t.1, t.2, t.3),
            Vec4::new(t.4, t.5, t.6, t.7),
            Vec4::new(t.8, t.9, t.10, t.11),
            Vec4::new(t.12, t.13, t.14, t.15),
        )
    }

    /// Creates a `Mat4` from a 2 dimensional `((f32, f32, f32, f32), (f32, f32, f32, f32),
    /// (f32, f32, f32, f32), (f32, f32, f32, f32))` tuple in **column-major** order.
    #[inline]
    #[must_use]
    pub fn from_2d_tuple(t: Mat4Tuple2D) -> Mat4 {
        Mat4::new(
            Vec4::new(t.0 .0, t.1 .0, t.2 .0, t.3 .0),
            Vec4::new(t.0 .1, t.1 .1, t.1 .2, t.1 .3),
            Vec4::new(t.0 .2, t.1 .2, t.2 .2, t.3 .2),
            Vec4::new(t.0 .3, t.1 .3, t.2 .3, t.3 .3),
        )
    }

    /// Converts `self` to a 16-element array in **row-major** order.
    #[inline]
    #[must_use]
    pub fn to_array_row_major(&self) -> [f32; 16] {
        [
            self.col0.x,
            self.col1.x,
            self.col2.x,
            self.col3.x,
            self.col0.y,
            self.col1.y,
            self.col2.y,
            self.col3.y,
            self.col0.z,
            self.col1.z,
            self.col2.z,
            self.col3.z,
            self.col0.w,
            self.col1.w,
            self.col2.w,
            self.col3.w,
        ]
    }

    /// Converts `self` to a 2 dimensional `[[f32; 4]; 4]` array in **row-major** order.
    #[inline]
    #[must_use]
    pub fn to_array_2d_row_major(&self) -> [[f32; 4]; 4] {
        [
            [self.col0.x, self.col1.x, self.col2.x, self.col3.x],
            [self.col0.y, self.col1.y, self.col2.y, self.col3.y],
            [self.col0.z, self.col1.z, self.col2.z, self.col3.z],
            [self.col0.w, self.col1.w, self.col2.w, self.col3.w],
        ]
    }

    /// Converts `self` to a 16-element array in **row-major** order.
    #[inline]
    #[must_use]
    pub fn to_array_col_major(&self) -> [f32; 16] {
        [
            self.col0.x,
            self.col0.y,
            self.col0.z,
            self.col0.w,
            self.col1.x,
            self.col1.y,
            self.col1.z,
            self.col1.w,
            self.col2.x,
            self.col2.y,
            self.col2.z,
            self.col2.w,
            self.col3.x,
            self.col3.y,
            self.col3.z,
            self.col3.w,
        ]
    }

    /// Converts `self` to a 2 dimensional `[[f32; 4]; 4]` array in **column-major** order.
    #[inline]
    #[must_use]
    pub fn to_array_2d_col_major(&self) -> [[f32; 4]; 4] {
        [
            [self.col0.x, self.col0.y, self.col0.z, self.col0.w],
            [self.col1.x, self.col1.y, self.col1.z, self.col1.w],
            [self.col2.z, self.col2.y, self.col2.z, self.col2.w],
            [self.col3.x, self.col3.y, self.col3.z, self.col3.w],
        ]
    }

    /// Converts `self` to a 16-element tuple in **row-major** order.
    #[inline]
    #[must_use]
    pub fn to_tuple_row_major(&self) -> Mat4Tuple {
        (
            self.col0.x,
            self.col1.x,
            self.col2.x,
            self.col3.x,
            self.col0.y,
            self.col1.y,
            self.col2.y,
            self.col3.y,
            self.col0.z,
            self.col1.z,
            self.col2.z,
            self.col3.z,
            self.col0.w,
            self.col1.w,
            self.col2.w,
            self.col3.w,
        )
    }

    /// Converts `self` to a 2 dimensional `((f32, f32, f32, f32), (f32, f32, f32, f32)
    /// (f32, f32, f32, f32), (f32, f32, f32, f32))` tuple in **row-major** order.
    #[inline]
    #[must_use]
    pub fn to_tuple_2d_row_major(&self) -> Mat4Tuple2D {
        (
            (self.col0.x, self.col1.x, self.col2.x, self.col3.x),
            (self.col0.y, self.col1.y, self.col2.y, self.col3.y),
            (self.col0.z, self.col1.z, self.col2.z, self.col3.z),
            (self.col0.w, self.col1.w, self.col2.w, self.col3.w),
        )
    }

    /// Converts `self` to a 16-element tuple in **column-major** order.
    #[inline]
    #[must_use]
    pub fn to_tuple_col_major(&self) -> Mat4Tuple {
        (
            self.col0.x,
            self.col0.y,
            self.col0.z,
            self.col0.w,
            self.col1.x,
            self.col1.y,
            self.col1.z,
            self.col1.w,
            self.col2.x,
            self.col2.y,
            self.col2.z,
            self.col2.w,
            self.col3.x,
            self.col3.y,
            self.col3.z,
            self.col3.w,
        )
    }

    /// Converts `self` to a 2 dimensional `((f32, f32, f32, f32), (f32, f32, f32, f32)
    /// (f32, f32, f32, f32), (f32, f32, f32, f32))` tuple in **column-major** order.
    #[inline]
    #[must_use]
    pub fn to_tuple_2d_col_major(&self) -> Mat4Tuple2D {
        (
            (self.col0.x, self.col0.y, self.col0.z, self.col0.w),
            (self.col1.x, self.col1.y, self.col1.z, self.col1.w),
            (self.col2.x, self.col2.y, self.col2.z, self.col2.w),
            (self.col3.x, self.col3.y, self.col3.z, self.col3.w),
        )
    }

    // ============= Constants ==============
    /// A 4x4 matrix with all elements set to zero.
    pub const ZERO: Self = Self {
        col0: Vec4::ZERO,
        col1: Vec4::ZERO,
        col2: Vec4::ZERO,
        col3: Vec4::ZERO,
    };

    /// The 4x4 identity matrix:
    ///
    /// ```text
    /// [1, 0, 0, 0]
    /// [0, 1, 0, 0]
    /// [0, 0, 1, 0]
    /// [0, 0, 0, 1]
    /// ```
    pub const IDENTITY: Self = Self {
        col0: Vec4 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        },
        col1: Vec4 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
            w: 0.0,
        },
        col2: Vec4 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
            w: 0.0,
        },
        col3: Vec4 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        },
    };

    /// A 4x4 matrix with all elements set to NaN.
    pub const NAN: Self = Self {
        col0: Vec4::NAN,
        col1: Vec4::NAN,
        col2: Vec4::NAN,
        col3: Vec4::NAN,
    };

    /// A 4x4 matrix with all elements set to infinity.
    pub const INFINITY: Self = Self {
        col0: Vec4::INFINITY,
        col1: Vec4::INFINITY,
        col2: Vec4::INFINITY,
        col3: Vec4::INFINITY,
    };

    // ============= Transformation Constructors ==============
    /// Creates a translation matrix
    #[inline]
    #[must_use]
    pub fn from_translation(translation: Vec3) -> Mat4 {
        Mat4::new(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(translation.x, translation.y, translation.z, 1.0),
        )
    }

    /// Creates a non-uniform scaling matrix.
    #[inline]
    #[must_use]
    pub fn from_scale(scale: Vec3) -> Mat4 {
        Mat4::new(
            Vec4::new(scale.x, 0.0, 0.0, 0.0),
            Vec4::new(0.0, scale.y, 0.0, 0.0),
            Vec4::new(0.0, 0.0, scale.z, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        )
    }

    /// Creates a rotation matrix from a quaternion.
    #[inline]
    #[must_use]
    pub fn from_quat(rotation: Quat) -> Mat4 {
        let (x2, y2, z2) = (
            rotation.x + rotation.x,
            rotation.y + rotation.y,
            rotation.z + rotation.z,
        );
        let (xx, xy, xz) = (rotation.x * x2, rotation.x * y2, rotation.x * z2);
        let (yy, yz, zz) = (rotation.y * y2, rotation.y * z2, rotation.z * z2);
        let (wx, wy, wz) = (rotation.w * x2, rotation.w * y2, rotation.w * z2);

        Self {
            col0: Vec4::new(1.0 - (yy + zz), xy + wz, xz - wy, 0.0),
            col1: Vec4::new(xy - wz, 1.0 - (xx + zz), yz + wx, 0.0),
            col2: Vec4::new(xz + wy, yz - wx, 1.0 - (xx + yy), 0.0),
            col3: Vec4::new(0.0, 0.0, 0.0, 1.0),
        }
    }

    /// Creates a rotation matrix around `x` axis.
    pub fn from_rotation_x(angle_rad: f32) -> Mat4 {
        let (sin, cos) = angle_rad.sin_cos();
        Mat4::new(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, cos, sin, 0.0),
            Vec4::new(0.0, -sin, cos, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        )
    }

    /// Creates a rotation matrix around `y` axis.
    pub fn from_rotation_y(angle_rad: f32) -> Mat4 {
        let (sin, cos) = angle_rad.sin_cos();
        Mat4::new(
            Vec4::new(cos, 0.0, -sin, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(sin, 0.0, cos, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        )
    }

    /// Creates a rotation matrix around `z` axis.
    pub fn from_rotation_z(angle_rad: f32) -> Mat4 {
        let (sin, cos) = angle_rad.sin_cos();
        Mat4::new(
            Vec4::new(cos, sin, 0.0, 0.0),
            Vec4::new(-sin, cos, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        )
    }

    /// Creates a transformation matrux from translation, rotation (quaternion), and scale components.
    /// The transformations are applied in the order: scale, then rotate, then translate.
    #[inline]
    #[must_use]
    pub fn from_trs(translation: Vec3, rotation: Quat, scale: Vec3) -> Mat4 {
        let rot_mat = Self::from_quat(rotation);
        Self {
            col0: rot_mat.col0 * scale.x,
            col1: rot_mat.col1 * scale.y,
            col2: rot_mat.col2 * scale.z,
            col3: translation.extend(1.0),
        }
    }

    /// Creates a right-handed perspective projection matrix.
    ///
    /// # Parameters
    /// - `fov_y_rad`: The vertical field of view in radians.
    /// - `aspect_ratio`: The aspect ratio of the viewport `(width / height)`.
    /// - `z_near`: The distance to the near clipping plane. Must be positive.
    /// - `z_far`: The distance to the far clipping plane. Must be positive.
    #[inline]
    #[must_use]
    pub fn perspective(fov_y_rad: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Mat4 {
        let f = 1.0 / (fov_y_rad / 2.0).tan();
        let nf = 1.0 / (z_near - z_far);
        Self {
            col0: Vec4::new(f / aspect_ratio, 0.0, 0.0, 0.0),
            col1: Vec4::new(0.0, f, 0.0, 0.0),
            col2: Vec4::new(0.0, 0.0, (z_far + z_near) * nf, -1.0),
            col3: Vec4::new(0.0, 0.0, 2.0 * z_far * z_near * nf, 0.0),
        }
    }

    /// Creates a right-handed orthographic projection matrix.
    #[inline]
    #[must_use]
    pub fn orthographic(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        z_near: f32,
        z_far: f32,
    ) -> Mat4 {
        let rml = 1.0 / (right - left);
        let tmb = 1.0 / (top - bottom);
        let fmn = 1.0 / (z_far - z_near);
        Self {
            col0: Vec4::new(2.0 * rml, 0.0, 0.0, 0.0),
            col1: Vec4::new(0.0, 2.0 * tmb, 0.0, 0.0),
            col2: Vec4::new(0.0, 0.0, -2.0 * fmn, 0.0),
            col3: Vec4::new(
                -(right + left) * rml,
                -(top + bottom) * tmb,
                -(z_far + z_near) * fmn,
                1.0,
            ),
        }
    }

    /// Creates a view matrix that looks from an `eye` position towards a `target` position.
    #[inline]
    #[must_use]
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
        Self::look_to(eye, target - eye, up)
    }

    /// Creates a view matrix that looks from an `eye` position in a specific `direction`.
    #[inline]
    #[must_use]
    pub fn look_to(eye: Vec3, direction: Vec3, up: Vec3) -> Mat4 {
        let f = direction.normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);
        Self {
            col0: Vec4::new(s.x, u.x, -f.x, 0.0),
            col1: Vec4::new(s.y, u.y, -f.y, 0.0),
            col2: Vec4::new(s.z, u.z, -f.z, 0.0),
            col3: Vec4::new(-eye.dot(s), -eye.dot(u), eye.dot(f), 1.0),
        }
    }

    /// Returns the transpose of the matrix.
    #[inline]
    #[must_use]
    pub fn transpose(&self) -> Self {
        Self {
            col0: Vec4::new(self.col0.x, self.col1.x, self.col2.x, self.col3.x),
            col1: Vec4::new(self.col0.y, self.col1.y, self.col2.y, self.col3.y),
            col2: Vec4::new(self.col0.z, self.col1.z, self.col2.z, self.col3.z),
            col3: Vec4::new(self.col0.w, self.col1.w, self.col2.w, self.col3.w),
        }
    }

    /// Computes the determinant of the matrix.
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> f32 {
        let c0 = self.col0;
        let c1 = self.col1;
        let c2 = self.col2;
        let c3 = self.col3;

        let a2323 = c2.z * c3.w - c2.w * c3.z;
        let a1323 = c2.y * c3.w - c2.w * c3.y;
        let a1223 = c2.y * c3.z - c2.z * c3.y;
        let a0323 = c2.x * c3.w - c2.w * c3.x;
        let a0223 = c2.x * c3.z - c2.z * c3.x;
        let a0123 = c2.x * c3.y - c2.y * c3.x;

        c0.x * (c1.y * a2323 - c1.z * a1323 + c1.w * a1223)
            - c0.y * (c1.x * a2323 - c1.z * a0323 + c1.w * a0223)
            + c0.z * (c1.x * a1323 - c1.y * a0323 + c1.w * a0123)
            - c0.w * (c1.x * a1223 - c1.y * a0223 + c1.z * a0123)
    }

    /// Computes the inverse of the matrix. Returns `None` if the matrix is not invertible.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let c0 = self.col0;
        let c1 = self.col1;
        let c2 = self.col2;
        let c3 = self.col3;

        let a0 = c0.x * c1.y - c0.y * c1.x;
        let a1 = c0.x * c1.z - c0.z * c1.x;
        let a2 = c0.x * c1.w - c0.w * c1.x;
        let a3 = c0.y * c1.z - c0.z * c1.y;
        let a4 = c0.y * c1.w - c0.w * c1.y;
        let a5 = c0.z * c1.w - c0.w * c1.z;

        let b0 = c2.x * c3.y - c2.y * c3.x;
        let b1 = c2.x * c3.z - c2.z * c3.x;
        let b2 = c2.x * c3.w - c2.w * c3.x;
        let b3 = c2.y * c3.z - c2.z * c3.y;
        let b4 = c2.y * c3.w - c2.w * c3.y;
        let b5 = c2.z * c3.w - c2.w * c3.z;

        let det = a0 * b5 - a1 * b4 + a2 * b3 + a3 * b2 - a4 * b1 + a5 * b0;

        if det.abs() < 1e-6 {
            return None;
        }

        let inv_det = 1.0 / det;

        let mut inv = Mat4::ZERO;
        inv.col0.x = (c1.y * b5 - c1.z * b4 + c1.w * b3) * inv_det;
        inv.col0.y = (-c0.y * b5 + c0.z * b4 - c0.w * b3) * inv_det;
        inv.col0.z = (c3.y * a5 - c3.z * a4 + c3.w * a3) * inv_det;
        inv.col0.w = (-c2.y * a5 + c2.z * a4 - c2.w * a3) * inv_det;
        inv.col1.x = (-c1.x * b5 + c1.z * b2 - c1.w * b1) * inv_det;
        inv.col1.y = (c0.x * b5 - c0.z * b2 + c0.w * b1) * inv_det;
        inv.col1.z = (-c3.x * a5 + c3.z * a2 - c3.w * a1) * inv_det;
        inv.col1.w = (c2.x * a5 - c2.z * a2 + c2.w * a1) * inv_det;
        inv.col2.x = (c1.x * b4 - c1.y * b2 + c1.w * b0) * inv_det;
        inv.col2.y = (-c0.x * b4 + c0.y * b2 - c0.w * b0) * inv_det;
        inv.col2.z = (c3.x * a4 - c3.y * a2 + c3.w * a0) * inv_det;
        inv.col2.w = (-c2.x * a4 + c2.y * a2 - c2.w * a0) * inv_det;
        inv.col3.x = (-c1.x * b3 + c1.y * b1 - c1.z * b0) * inv_det;
        inv.col3.y = (c0.x * b3 - c0.y * b1 + c0.z * b0) * inv_det;
        inv.col3.z = (-c3.x * a3 + c3.y * a1 - c3.z * a0) * inv_det;
        inv.col3.w = (c2.x * a3 - c2.y * a1 + c2.z * a0) * inv_det;

        Some(inv)
    }

    /// Checks if the matrix is invertible (determinant != 0)
    #[inline]
    #[must_use]
    pub fn is_invertible(&self) -> bool {
        self.determinant().abs() >= 1e-6
    }

    /// Checks if 2 matrices are approximately equal within a default epsilon.
    pub fn approx_eq(&self, other: Mat4) -> bool {
        self.col0.approx_eq(other.col0)
            && self.col1.approx_eq(other.col1)
            && self.col2.approx_eq(other.col2)
            && self.col3.approx_eq(other.col3)
    }

    /// Checks if 2 matrices are approximately equal within a given `epsilon`.
    pub fn approx_eq_eps(&self, other: Mat4, epsilon: f32) -> bool {
        self.col0.approx_eq_eps(other.col0, epsilon)
            && self.col1.approx_eq_eps(other.col1, epsilon)
            && self.col2.approx_eq_eps(other.col2, epsilon)
            && self.col3.approx_eq_eps(other.col3, epsilon)
    }

    /// Transforms a 3D point, including perspective division.
    #[inline]
    #[must_use]
    pub fn project_point3(&self, point: Vec3) -> Vec3 {
        let mut res = *self * point.extend(1.0);
        res /= res.w;
        res.xyz()
    }

    /// Transforms a 3D point by an affine matrix (w=1).
    #[inline]
    #[must_use]
    pub fn transform_point3(&self, point: Vec3) -> Vec3 {
        (*self * point.extend(1.0)).xyz()
    }

    /// Transforms a 3D vector (direction, w=0)
    #[inline]
    #[must_use]
    pub fn transform_vector3(&self, vector: Vec3) -> Vec3 {
        (*self * vector.extend(0.0)).xyz()
    }

    /// Returns true if all elements are finite (not NaN or infinity)
    pub fn is_finite(self) -> bool {
        self.col0.is_finite()
            && self.col1.is_finite()
            && self.col2.is_finite()
            && self.col3.is_finite()
    }

    /// Returns true if any elements are NaN
    pub fn is_nan(self) -> bool {
        self.col0.is_nan() || self.col1.is_nan() || self.col2.is_nan() || self.col3.is_nan()
    }

    /// Adjugates the matrix
    pub fn adjugate(&self) -> Self {
        // Temporary variables for better readability
        let a = self.col0;
        let b = self.col1;
        let c = self.col2;
        let d = self.col3;

        // Cofactor computations (3x3 determinants with sign alternation)
        Mat4::new(
            Vec4::new(
                b.y * (c.z * d.w - c.w * d.z) - b.z * (c.y * d.w - c.w * d.y)
                    + b.w * (c.y * d.z - c.z * d.y), // (0,0)
                -a.y * (c.z * d.w - c.w * d.z) + a.z * (c.y * d.w - c.w * d.y)
                    - a.w * (c.y * d.z - c.z * d.y), // (1,0)
                a.y * (b.z * d.w - b.w * d.z) - a.z * (b.y * d.w - b.w * d.y)
                    + a.w * (b.y * d.z - b.z * d.y), // (2,0)
                -a.y * (b.z * c.w - b.w * c.z) + a.z * (b.y * c.w - b.w * c.y)
                    - a.w * (b.y * c.z - b.z * c.y), // (3,0)
            ),
            Vec4::new(
                -b.x * (c.z * d.w - c.w * d.z) + b.z * (c.x * d.w - c.w * d.x)
                    - b.w * (c.x * d.z - c.z * d.x), // (0,1)
                a.x * (c.z * d.w - c.w * d.z) - a.z * (c.x * d.w - c.w * d.x)
                    + a.w * (c.x * d.z - c.z * d.x), // (1,1)
                -a.x * (b.z * d.w - b.w * d.z) + a.z * (b.x * d.w - b.w * d.x)
                    - a.w * (b.x * d.z - b.z * d.x), // (2,1)
                a.x * (b.z * c.w - b.w * c.z) - a.z * (b.x * c.w - b.w * c.x)
                    + a.w * (b.x * c.z - b.z * c.x), // (3,1)
            ),
            Vec4::new(
                b.x * (c.y * d.w - c.w * d.y) - b.y * (c.x * d.w - c.w * d.x)
                    + b.w * (c.x * d.y - c.y * d.x), // (0,2)
                -a.x * (c.y * d.w - c.w * d.y) + a.y * (c.x * d.w - c.w * d.x)
                    - a.w * (c.x * d.y - c.y * d.x), // (1,2)
                a.x * (b.y * d.w - b.w * d.y) - a.y * (b.x * d.w - b.w * d.x)
                    + a.w * (b.x * d.y - b.y * d.x), // (2,2)
                -a.x * (b.y * c.w - b.w * c.y) + a.y * (b.x * c.w - b.w * c.x)
                    - a.w * (b.x * c.y - b.y * c.x), // (3,2)
            ),
            Vec4::new(
                -b.x * (c.y * d.z - c.z * d.y) + b.y * (c.x * d.z - c.z * d.x)
                    - b.z * (c.x * d.y - c.y * d.x), // (0,3)
                a.x * (c.y * d.z - c.z * d.y) - a.y * (c.x * d.z - c.z * d.x)
                    + a.z * (c.x * d.y - c.y * d.x), // (1,3)
                -a.x * (b.y * d.z - b.z * d.y) + a.y * (b.x * d.z - b.z * d.x)
                    - a.z * (b.x * d.y - b.y * d.x), // (2,3)
                a.x * (b.y * c.z - b.z * c.y) - a.y * (b.x * c.z - b.z * c.x)
                    + a.z * (b.x * c.y - b.y * c.x), // (3,3)
            ),
        )
    }

    ///Returns the sum of the diagonal elements
    pub fn trace(self) -> f32 {
        self.col0.x + self.col1.y + self.col2.z + self.col3.w
    }
}

// ============= Operator Overloads =============

/// Adds two matrices together component-wise.
impl Add for Mat4 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.col0 + rhs.col0,
            self.col1 + rhs.col1,
            self.col2 + rhs.col2,
            self.col3 + rhs.col3,
        )
    }
}

/// Subtracts `rhs` from `self` component-wise.
impl Sub for Mat4 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.col0 - rhs.col0,
            self.col1 - rhs.col1,
            self.col2 - rhs.col2,
            self.col3 - rhs.col3,
        )
    }
}

/// Multiplies two matrices using standard matrix multiplication.
impl Mul for Mat4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self * rhs.col0,
            self * rhs.col1,
            self * rhs.col2,
            self * rhs.col3,
        )
    }
}

/// Multiplies the matrix by a `Vec4` (matrix-vector multiplication).
impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline]
    fn mul(self, rhs: Vec4) -> Self::Output {
        self.col0 * rhs.x + self.col1 * rhs.y + self.col2 * rhs.z + self.col3 * rhs.w
    }
}

/// Multiplies each component of the matrix by a scalar.
impl Mul<f32> for Mat4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(
            self.col0 * rhs,
            self.col1 * rhs,
            self.col2 * rhs,
            self.col3 * rhs,
        )
    }
}

/// Multiplies a scalar by each component of the matrix.
impl Mul<Mat4> for f32 {
    type Output = Mat4;
    #[inline]
    fn mul(self, rhs: Mat4) -> Self::Output {
        rhs * self
    }
}

/// Divides each component of the matrix by a scalar.
impl Div<f32> for Mat4 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Self::new(
            self.col0 / rhs,
            self.col1 / rhs,
            self.col2 / rhs,
            self.col3 / rhs,
        )
    }
}

/// Negates each component of the matrix.
impl Neg for Mat4 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.col0, -self.col1, -self.col2, -self.col3)
    }
}

// ============= Assignment Operator Overloads =============

impl AddAssign for Mat4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.col0 += rhs.col0;
        self.col1 += rhs.col1;
        self.col2 += rhs.col2;
        self.col3 += rhs.col3;
    }
}

impl SubAssign for Mat4 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.col0 -= rhs.col0;
        self.col1 -= rhs.col1;
        self.col2 -= rhs.col2;
        self.col3 -= rhs.col3;
    }
}

impl MulAssign for Mat4 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<f32> for Mat4 {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.col0 *= rhs;
        self.col1 *= rhs;
        self.col2 *= rhs;
        self.col3 *= rhs;
    }
}

impl DivAssign<f32> for Mat4 {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.col0 /= rhs;
        self.col1 /= rhs;
        self.col2 /= rhs;
        self.col3 /= rhs;
    }
}

// ============= Trait Implementations =============

impl Default for Mat4 {
    /// Returns the identity matrix.
    #[inline]
    fn default() -> Self {
        Self::IDENTITY // Assumes Mat4::IDENTITY constant exists
    }
}

/// Checks whether two matrices are exactly equal.
impl PartialEq for Mat4 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.col0 == other.col0
            && self.col1 == other.col1
            && self.col2 == other.col2
            && self.col3 == other.col3
    }
}

/// Enables `m[column]` access. Panics if `col_index` is out of bounds.
impl Index<usize> for Mat4 {
    type Output = Vec4;
    #[inline]
    fn index(&self, col_index: usize) -> &Self::Output {
        match col_index {
            0 => &self.col0,
            1 => &self.col1,
            2 => &self.col2,
            3 => &self.col3,
            _ => panic!("Mat4 column index out of bounds: {}", col_index),
        }
    }
}

/// Enables mutable `m[column]` access. Panics if `col_index` is out of bounds.
impl IndexMut<usize> for Mat4 {
    #[inline]
    fn index_mut(&mut self, col_index: usize) -> &mut Self::Output {
        match col_index {
            0 => &mut self.col0,
            1 => &mut self.col1,
            2 => &mut self.col2,
            3 => &mut self.col3,
            _ => panic!("Mat4 column index out of bounds: {}", col_index),
        }
    }
}

/// Implements the `Display` trait for pretty-printing.
impl fmt::Display for Mat4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:.3}, {:.3}, {:.3}, {:.3}]\n[{:.3}, {:.3}, {:.3}, {:.3}]\n[{:.3}, {:.3}, {:.3}, {:.3}]\n[{:.3}, {:.3}, {:.3}, {:.3}]",
            self.col0.x,
            self.col1.x,
            self.col2.x,
            self.col3.x,
            self.col0.y,
            self.col1.y,
            self.col2.y,
            self.col3.y,
            self.col0.z,
            self.col1.z,
            self.col2.z,
            self.col3.z,
            self.col0.w,
            self.col1.w,
            self.col2.w,
            self.col3.w
        )
    }
}

// ============= Approx Crate Implementations =============

/// Implements absolute difference equality comparison for `Mat4`.
impl approx::AbsDiffEq for Mat4 {
    type Epsilon = f32;

    #[inline]
    fn default_epsilon() -> f32 {
        f32::EPSILON
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: f32) -> bool {
        self.col0.abs_diff_eq(&other.col0, epsilon)
            && self.col1.abs_diff_eq(&other.col1, epsilon)
            && self.col2.abs_diff_eq(&other.col2, epsilon)
            && self.col3.abs_diff_eq(&other.col3, epsilon)
    }
}

/// Implements relative equality comparison for `Mat4`.
impl approx::RelativeEq for Mat4 {
    #[inline]
    fn default_max_relative() -> f32 {
        f32::EPSILON
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: f32, max_relative: f32) -> bool {
        self.col0.relative_eq(&other.col0, epsilon, max_relative)
            && self.col1.relative_eq(&other.col1, epsilon, max_relative)
            && self.col2.relative_eq(&other.col2, epsilon, max_relative)
            && self.col3.relative_eq(&other.col3, epsilon, max_relative)
    }
}
