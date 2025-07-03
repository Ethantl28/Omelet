use crate::vec::Vec4;
use crate::vec::Vec3;
use std::ops::{Add, Sub, Mul, Div};

///4x4 column-major matrix for 3D transformations
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub x: Vec4,
    pub y: Vec4,
    pub z: Vec4,
    pub w: Vec4,
}

impl Mat4 {
    ///Creates a new matrix from column vectors
    pub fn new(x: Vec4, y: Vec4, z: Vec4, w: Vec4) -> Mat4 {
        Mat4{x, y, z, w}
    }

    ///Creates an identity matrix (1.0 on diagonal)
    pub fn identity() -> Mat4 {
        Mat4::new(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0)
        )
    }

    ///Creates a zero matrix
    pub fn zero() -> Mat4 {
        Mat4::new(
            Vec4::zero(),
            Vec4::zero(),
            Vec4::zero(),
            Vec4::zero()
        )
    }

    ///Creates a diagonal matrix
    pub fn from_diagonal(diag: Vec3) -> Mat4 {
        Mat4::new(
            Vec4::new(diag.x, 0.0, 0.0, 0.0),
            Vec4::new(0.0, diag.y, 0.0, 0.0),
            Vec4::new(0.0, 0.0, diag.z, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0)
        )
    }

    ///Creates a translation matrix
    pub fn from_translation(translation: Vec3) -> Mat4 {
        Mat4::new(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(translation.x, translation.y, translation.z, 1.0)
        )
    }

    ///Creates a rotation matrix from x
    pub fn from_rotation_x(angle_rad: f32) -> Mat4 {
        let (sin, cos) = angle_rad.sin_cos();
        Mat4::new(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, cos, sin, 0.0),
            Vec4::new(0.0, -sin, cos, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0)
        )
    }

    ///Creates a rotation matrix from y
    pub fn from_rotation_y(angle_rad: f32) -> Mat4 {
        let (sin, cos) = angle_rad.sin_cos();
        Mat4::new(
            Vec4::new(cos, 0.0, -sin, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(sin, 0.0, cos, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0)
        )
    }

    ///Creates a rotation matrix from z
    pub fn from_rotation_z(angle_rad: f32) -> Mat4 {
        let (sin, cos) = angle_rad.sin_cos();
        Mat4::new(
            Vec4::new(cos, sin, 0.0, 0.0),
            Vec4::new(-sin, cos, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0)
        )
    }

    ///Creates a scaling matrix
    pub fn from_scale(scale: Vec3) -> Mat4 {
        Mat4::from_diagonal(scale)
    }

    ///Creates a perspective matrix
    pub fn from_perspective(fov_rad: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let tan_half_fov = (fov_rad / 2.0).tan();
        let range_inv = 1.0 / (near - far);

        Mat4::new(
            Vec4::new(1.0 / (aspect * tan_half_fov), 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0 / tan_half_fov, 0.0, 0.0),
            Vec4::new(0.0, 0.0, (near + far) * range_inv, -1.0),
            Vec4::new(0.0, 0.0, 2.0 * near * far * range_inv, 0.0)
    )
}

    ///Creates an orthographic matrix
    pub fn from_orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat4 {
        let width = right - left;
        let height = top - bottom;
        let depth = far - near;

         Self::new(
            Vec4::new(2.0 / width, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0 / height, 0.0, 0.0),
            Vec4::new(0.0, 0.0, -2.0 / depth, 0.0),
            Vec4::new(
                -(right + left) / width,
                -(top + bottom) / height,
                -(far + near) / depth,
                1.0
            )
        )
    }

    ///Computes the inverse of the matrix if it exists (determinant != 0)
    ///Returns None if the matrix is singular
    pub fn inverse(&self) -> Option<Mat4> {
        let det = self.determinant();
        if det.abs() < f32::EPSILON {
            return None;
        }

        let inv_det = 1.0 / det;

        let m = self.transpose(); // We use transpose for cofactor calculation
        let c00 = m.y.y * m.z.z * m.w.w + m.y.z * m.z.w * m.w.y + m.y.w * m.z.y * m.w.z - m.y.y * m.z.w * m.w.z - m.y.z * m.z.y * m.w.w - m.y.w * m.z.z * m.w.y;
        let c01 = m.y.x * m.z.w * m.w.z + m.y.z * m.z.x * m.w.w + m.y.w * m.z.z * m.w.x - m.y.x * m.z.z * m.w.w - m.y.z * m.z.w * m.w.x - m.y.w * m.z.x * m.w.z;
        let c02 = m.y.x * m.z.y * m.w.w + m.y.y * m.z.w * m.w.x + m.y.w * m.z.x * m.w.y - m.y.x * m.z.w * m.w.y - m.y.y * m.z.x * m.w.w - m.y.w * m.z.y * m.w.x;
        let c03 = m.y.x * m.z.z * m.w.y + m.y.y * m.z.x * m.w.z + m.y.z * m.z.y * m.w.x - m.y.x * m.z.y * m.w.z - m.y.y * m.z.z * m.w.x - m.y.z * m.z.x * m.w.y;

        let c10 = m.x.y * m.z.w * m.w.z + m.x.z * m.z.y * m.w.w + m.x.w * m.z.z * m.w.y - m.x.y * m.z.z * m.w.w - m.x.z * m.z.w * m.w.y - m.x.w * m.z.y * m.w.z;
        let c11 = m.x.x * m.z.z * m.w.w + m.x.z * m.z.w * m.w.x + m.x.w * m.z.x * m.w.z - m.x.x * m.z.w * m.w.z - m.x.z * m.z.x * m.w.w - m.x.w * m.z.z * m.w.x;
        let c12 = m.x.x * m.z.w * m.w.y + m.x.y * m.z.x * m.w.w + m.x.w * m.z.y * m.w.x - m.x.x * m.z.y * m.w.w - m.x.y * m.z.w * m.w.x - m.x.w * m.z.x * m.w.y;
        let c13 = m.x.x * m.z.y * m.w.z + m.x.y * m.z.z * m.w.x + m.x.z * m.z.x * m.w.y - m.x.x * m.z.z * m.w.y - m.x.y * m.z.x * m.w.z - m.x.z * m.z.y * m.w.x;

        let c20 = m.x.y * m.y.z * m.w.w + m.x.z * m.y.w * m.w.y + m.x.w * m.y.y * m.w.z - m.x.y * m.y.w * m.w.z - m.x.z * m.y.y * m.w.w - m.x.w * m.y.z * m.w.y;
        let c21 = m.x.x * m.y.w * m.w.z + m.x.z * m.y.x * m.w.w + m.x.w * m.y.z * m.w.x - m.x.x * m.y.z * m.w.w - m.x.z * m.y.w * m.w.x - m.x.w * m.y.x * m.w.z;
        let c22 = m.x.x * m.y.y * m.w.w + m.x.y * m.y.w * m.w.x + m.x.w * m.y.x * m.w.y - m.x.x * m.y.w * m.w.y - m.x.y * m.y.x * m.w.w - m.x.w * m.y.y * m.w.x;
        let c23 = m.x.x * m.y.z * m.w.y + m.x.y * m.y.x * m.w.z + m.x.z * m.y.y * m.w.x - m.x.x * m.y.y * m.w.z - m.x.y * m.y.z * m.w.x - m.x.z * m.y.x * m.w.y;

        let c30 = m.x.y * m.y.w * m.z.z + m.x.z * m.y.y * m.z.w + m.x.w * m.y.z * m.z.y - m.x.y * m.y.z * m.z.w - m.x.z * m.y.w * m.z.y - m.x.w * m.y.y * m.z.z;
        let c31 = m.x.x * m.y.z * m.z.w + m.x.z * m.y.w * m.z.x + m.x.w * m.y.x * m.z.z - m.x.x * m.y.w * m.z.z - m.x.z * m.y.x * m.z.w - m.x.w * m.y.z * m.z.x;
        let c32 = m.x.x * m.y.w * m.z.y + m.x.y * m.y.x * m.z.w + m.x.w * m.y.y * m.z.x - m.x.x * m.y.y * m.z.w - m.x.y * m.y.w * m.z.x - m.x.w * m.y.x * m.z.y;
        let c33 = m.x.x * m.y.y * m.z.z + m.x.y * m.y.z * m.z.x + m.x.z * m.y.x * m.z.y - m.x.x * m.y.z * m.z.y - m.x.y * m.y.x * m.z.z - m.x.z * m.y.y * m.z.x;

        let adjugate = Mat4::new(
            Vec4::new(c00, c01, c02, c03),
            Vec4::new(c10, c11, c12, c13),
            Vec4::new(c20, c21, c22, c23),
            Vec4::new(c30, c31, c32, c33),
        );

        Some(adjugate * inv_det)
    }

    ///Checks if the matrix is invertible (determinant != 0)
    pub fn is_invertible(&self) -> bool {
        self.determinant().abs() >= f32::EPSILON
    }

    ///Transposes matrix
    pub fn transpose(&self) -> Mat4 {
        Mat4::new(
            Vec4::new(self.x.x, self.y.x, self.z.x, self.w.x),
            Vec4::new(self.x.y, self.y.y, self.z.y, self.w.y),
            Vec4::new(self.x.z, self.y.z, self.z.z, self.w.z),
            Vec4::new(self.x.w, self.y.w, self.z.w, self.w.w)
        )
    }

    ///Calculates the determinant
    pub fn determinant(&self) -> f32 {
        let m = self;
        m.x.x * (m.y.y * (m.z.z * m.w.w - m.z.w * m.w.z) - m.y.z * (m.z.y * m.w.w - m.z.w * m.w.y) + m.y.w * (m.z.y * m.w.z - m.z.z * m.w.y)) -
        m.x.y * (m.y.x * (m.z.z * m.w.w - m.z.w * m.w.z) - m.y.z * (m.z.x * m.w.w - m.z.w * m.w.x) + m.y.w * (m.z.x * m.w.z - m.z.z * m.w.x)) +
        m.x.z * (m.y.x * (m.z.y * m.w.w - m.z.w * m.w.y) - m.y.y * (m.z.x * m.w.w - m.z.w * m.w.x) + m.y.w * (m.z.x * m.w.y - m.z.y * m.w.x)) -
        m.x.w * (m.y.x * (m.z.y * m.w.z - m.z.z * m.w.y) - m.y.y * (m.z.x * m.w.z - m.z.z * m.w.x) + m.y.z * (m.z.x * m.w.y - m.z.y * m.w.x))
}

    ///Checks if matrices are approx equal
    pub fn approx_eq(&self, other: Mat4) -> bool {
        self.x.approx_eq(other.x) && self.y.approx_eq(other.y) && self.z.approx_eq(other.z)
    }

    ///Transforms point
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        let v = *self * Vec4::new(point.x, point.y, point.z, 1.0);
        Vec3::new(v.x / v.w, v.y / v.w, v.z / v.w) // Perspective divide
    }

    ///Transforms vector
    pub fn transform_vector(&self, vector: Vec3) -> Vec3 {
        let v = *self * Vec4::new(vector.x, vector.y, vector.z, 0.0);
        Vec3::new(v.x, v.y, v.z)
    }

    ///Returns true if all elements are finite (not NaN or infinity)
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }

    ///Returns true if any elements are NaN
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan() || self.w.is_nan()
    }

    ///Adjugates the matrix
    pub fn adjugate(&self) -> Self {
        // Temporary variables for better readability
        let a = self.x; let b = self.y;
        let c = self.z; let d = self.w;

        // Cofactor computations (3x3 determinants with sign alternation)
        Mat4::new(
            Vec4::new(
                b.y*(c.z*d.w - c.w*d.z) - b.z*(c.y*d.w - c.w*d.y) + b.w*(c.y*d.z - c.z*d.y),  // (0,0)
                -a.y*(c.z*d.w - c.w*d.z) + a.z*(c.y*d.w - c.w*d.y) - a.w*(c.y*d.z - c.z*d.y), // (1,0)
                a.y*(b.z*d.w - b.w*d.z) - a.z*(b.y*d.w - b.w*d.y) + a.w*(b.y*d.z - b.z*d.y),  // (2,0)
                -a.y*(b.z*c.w - b.w*c.z) + a.z*(b.y*c.w - b.w*c.y) - a.w*(b.y*c.z - b.z*c.y)  // (3,0)
            ),
            Vec4::new(
                -b.x*(c.z*d.w - c.w*d.z) + b.z*(c.x*d.w - c.w*d.x) - b.w*(c.x*d.z - c.z*d.x), // (0,1)
                a.x*(c.z*d.w - c.w*d.z) - a.z*(c.x*d.w - c.w*d.x) + a.w*(c.x*d.z - c.z*d.x),  // (1,1)
                -a.x*(b.z*d.w - b.w*d.z) + a.z*(b.x*d.w - b.w*d.x) - a.w*(b.x*d.z - b.z*d.x), // (2,1)
                a.x*(b.z*c.w - b.w*c.z) - a.z*(b.x*c.w - b.w*c.x) + a.w*(b.x*c.z - b.z*c.x)   // (3,1)
            ),
            Vec4::new(
                b.x*(c.y*d.w - c.w*d.y) - b.y*(c.x*d.w - c.w*d.x) + b.w*(c.x*d.y - c.y*d.x),  // (0,2)
                -a.x*(c.y*d.w - c.w*d.y) + a.y*(c.x*d.w - c.w*d.x) - a.w*(c.x*d.y - c.y*d.x), // (1,2)
                a.x*(b.y*d.w - b.w*d.y) - a.y*(b.x*d.w - b.w*d.x) + a.w*(b.x*d.y - b.y*d.x),  // (2,2)
                -a.x*(b.y*c.w - b.w*c.y) + a.y*(b.x*c.w - b.w*c.x) - a.w*(b.x*c.y - b.y*c.x)  // (3,2)
            ),
            Vec4::new(
                -b.x*(c.y*d.z - c.z*d.y) + b.y*(c.x*d.z - c.z*d.x) - b.z*(c.x*d.y - c.y*d.x), // (0,3)
                a.x*(c.y*d.z - c.z*d.y) - a.y*(c.x*d.z - c.z*d.x) + a.z*(c.x*d.y - c.y*d.x),  // (1,3)
                -a.x*(b.y*d.z - b.z*d.y) + a.y*(b.x*d.z - b.z*d.x) - a.z*(b.x*d.y - b.y*d.x), // (2,3)
                a.x*(b.y*c.z - b.z*c.y) - a.y*(b.x*c.z - b.z*c.x) + a.z*(b.x*c.y - b.y*c.x)   // (3,3)
            )
        )
    }

    ///Returns the sum of the diagonal elements
    pub fn trace(self) -> f32 {
        self.x.x + self.y.y + self.z.z + self.w.w
    }

    ///Swaps rows of matrix
    /// 
    ///# Panics
    /// Panics if row_a or row_b >= 4
    pub fn swap_rows(&mut self, row_a: usize, row_b: usize) {
        assert!(row_a < 4 && row_b < 4, "Row indices must be 0-3");
        if row_a == row_b { return; } //No point in continuing swap

        //Uses SIMD optimization where available
        unsafe {
            let ptr = self as *mut Mat4 as *mut [f32; 16];
            let slice = &mut *ptr;
            for i in 0..4 {
                slice.swap(row_a * 4 + i, row_b * 4 + i);
            }
        }
    }
}

//Operator overloads
impl Add for Mat4 {
    type Output = Self;
    fn add(self, rhs: Mat4) -> Mat4 {
        Mat4::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z, self.w + rhs.w)
    }
}

impl Sub for Mat4 {
    type Output = Self;
    fn sub(self, rhs: Mat4) -> Mat4 {
        Mat4::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z, self.w - rhs.w)
    }
}

impl Mul for Mat4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self * (rhs.x),
            self * (rhs.y),
            self * (rhs.z),
            self * (rhs.w)
        )
    }
}

impl Mul<f32> for Mat4 {
    type Output = Self;
    fn mul(self, scalar: f32) -> Self {
        Self::new(self.x * scalar, self.y * scalar, self.z * scalar, self.w * scalar)
    }
}


impl Mul<Mat4> for f32 {
    type Output = Mat4;
    fn mul(self, mat: Mat4) -> Mat4 {
        mat * self
    }
}

impl Div<f32> for Mat4 {
    type Output = Self;
    fn div(self, scalar: f32) -> Self {
        Mat4::new(self.x / scalar, self.y / scalar, self.z / scalar, self.w / scalar)
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    fn mul(self, v: Vec4) -> Vec4 {
        Vec4::new(
            self.x.x * v.x + self.y.x * v.y + self.z.x * v.z + self.w.x * v.w,
            self.x.y * v.x + self.y.y * v.y + self.z.y * v.z + self.w.y * v.w,
            self.x.z * v.x + self.y.z * v.y + self.z.z * v.z + self.w.z * v.w,
            self.x.w * v.x + self.y.w * v.y + self.z.w * v.z + self.w.w * v.w
        )
    }
}

impl Default for Mat4 {
    fn default() -> Self {
        Mat4{
            x: Vec4::zero(),
            y: Vec4::zero(),
            z: Vec4::zero(),
            w: Vec4::zero(),
        }
    }
}

use std::ops::{Index, IndexMut};
impl Index<usize> for Mat4 {
    type Output = Vec4;
    
    fn index(&self, row: usize) -> &Vec4 {
        match row {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Mat4 row index out of bounds: {}", row),
        }
    }
}

impl IndexMut<usize> for Mat4 {
    fn index_mut(&mut self, row: usize) -> &mut Vec4 {
        match row {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Mat4 row index out of bounds: {}", row),
        }
    }
}