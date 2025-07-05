use crate::mat4::Mat4;
use crate::utils::epsilon_eq_default;
use crate::vec::Vec2;
use crate::vec::Vec3;
use crate::vec::Vec4;
use core::f32;
use std::ops::{Add, Div, Mul, Sub};

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
        Mat3 { x, y, z }
    }

    ///Creates an identity matrix (1.0 on diagonal)
    pub fn identity() -> Mat3 {
        Mat3::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
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
            Vec3::new(translation.x, translation.y, 1.0),
        )
    }

    ///Creates a rotation matrix (angle in radians)
    pub fn from_rotation_z(angle_rad: f32) -> Mat3 {
        let (sin, cos) = angle_rad.sin_cos();
        Mat3::new(
            Vec3::new(cos, sin, 0.0),
            Vec3::new(-sin, cos, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
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
            Vec3::new(self.x.z, self.y.z, self.z.z),
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
            ),
        ))
    }

    ///Checks if matrices are approx equal using default epsilon
    pub fn approx_eq(&self, other: Mat3) -> bool {
        self.x.approx_eq(other.x) && self.y.approx_eq(other.y) && self.z.approx_eq(other.z)
    }

    ///Checks if matrices are approx equal using user entered epsilon
    pub fn approx_eq_eps(&self, other: Mat3, epsilon: f32) -> bool {
        self.x.approx_eq_eps(other.x, epsilon)
            && self.y.approx_eq_eps(other.y, epsilon)
            && self.z.approx_eq_eps(other.z, epsilon)
    }

    ///Transforms point
    pub fn transform_point(&self, point: Vec2) -> Vec2 {
        let v = *self * (Vec3::new(point.x, point.y, 1.0));
        Vec2::new(v.x, v.y)
    }

    ///Transforms vector
    pub fn transform_vector(&self, vector: Vec2) -> Vec2 {
        let v = *self * (Vec3::new(vector.x, vector.y, 0.0));
        Vec2::new(v.x, v.y)
    }

    ///Returns true if all elements are finite (not NaN or infinity)
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    ///Returns true if any elements are NaN
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }

    ///Adjugates the matrix
    pub fn adjugate(&self) -> Self {
        let m = self;
        Mat3::new(
            Vec3::new(
                m.y.y * m.z.z - m.y.z * m.z.y,
                m.x.z * m.z.y - m.x.y * m.z.z,
                m.x.y * m.y.z - m.x.z * m.y.y,
            ),
            Vec3::new(
                m.y.z * m.z.x - m.y.x * m.z.z,
                m.x.x * m.z.z - m.x.z * m.z.x,
                m.x.z * m.y.x - m.x.x * m.y.z,
            ),
            Vec3::new(
                m.y.x * m.z.y - m.y.y * m.z.x,
                m.x.y * m.z.x - m.x.x * m.z.y,
                m.x.x * m.y.y - m.x.y * m.y.x,
            ),
        )
    }

    ///Returns the sum of the diagonal elements
    pub fn trace(self) -> f32 {
        self.x.x + self.y.y + self.z.z
    }

    ///Returns translation from the matrix
    pub fn get_translation(&self) -> Vec2 {
        Vec2::new(self.z.x, self.z.y)
    }

    ///Returns the scale from the matrix
    pub fn get_scale(&self) -> Vec2 {
        Vec2::new(
            Vec2::new(self.x.x, self.x.y).length(),
            Vec2::new(self.y.x, self.y.y).length(),
        )
    }

    ///Returns the rotation from the matrix
    pub fn get_rotation(&self) -> f32 {
        //Extract the x basis vector
        let x_basis = Vec2::new(self.x.x, self.x.y);

        //Normalize to remove scale
        let normalized_x = x_basis.normalize();

        //Calculate angle using atan2
        normalized_x.y.atan2(normalized_x.x)
    }

    ///Returns translation, rotation, and scale from the matrix
    pub fn decompose(&self) -> (Vec2, f32, Vec2) {
        (
            self.get_translation(),
            self.get_rotation(),
            self.get_scale(),
        )
    }

    ///Creates a 2D view matrix looking from eye to target
    pub fn look_at(eye: Vec2, target: Vec2) -> Mat3 {
        let forward = (target - eye).normalize();
        let right = forward.perpendicular();

        Mat3::new(
            Vec3::new(right.x, right.y, 0.0),
            Vec3::new(forward.x, forward.y, 0.0),
            Vec3::new(-right.dot(eye), -forward.dot(eye), 1.0),
        )
    }

    ///Creates an orthographic projection matrix for 2D
    pub fn ortho(left: f32, right: f32, bottom: f32, top: f32) -> Mat3 {
        let tx = -(right + left) / (right - left);
        let ty = (top + bottom) / (top - bottom);

        Mat3::new(
            Vec3::new(2.0 / (right - left), 0.0, 0.0),
            Vec3::new(0.0, 2.0 / (top - bottom), 0.0),
            Vec3::new(tx, ty, 1.0),
        )
    }

    ///Linearly interpolated between two matrices
    pub fn lerp(a: Mat3, b: Mat3, t: f32) -> Mat3 {
        Mat3::new(
            Vec3::lerp(a.x, b.x, t),
            Vec3::lerp(a.y, b.y, t),
            Vec3::lerp(a.z, b.z, t),
        )
    }

    ///Creates a shear matrix
    pub fn from_shear(shear: Vec2) -> Mat3 {
        Mat3::new(
            Vec3::new(1.0, shear.y, 0.0),
            Vec3::new(shear.x, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
    }

    ///Builder method for translation  (consuming)
    pub fn with_translation(self, translation: Vec2) -> Mat3 {
        Mat3::from_translation(translation) * self
    }

    ///Builder method for rotation (consuming)
    pub fn with_rotation(self, angle: f32) -> Mat3 {
        Mat3::from_rotation_z(angle) * self
    }

    ///Builder method for translation (non consuming)
    pub fn apply_translation(&mut self, translation: Vec2) {
        *self = Mat3::from_translation(translation) * *self
    }

    ///Builer method for rotation (non consuming)
    pub fn apply_rotation(&mut self, angle: f32) {
        *self = Mat3::from_rotation_z(angle)
    }

    ///Transforms a 2D AABB and returns a new AABB
    pub fn transform_aabb(&self, min: Vec2, max: Vec2) -> (Vec2, Vec2) {
        let corners = [
            self.transform_point(Vec2::new(min.x, min.y)),
            self.transform_point(Vec2::new(min.x, max.y)),
            self.transform_point(Vec2::new(max.x, min.y)),
            self.transform_point(Vec2::new(max.x, max.y)),
        ];

        let mut new_min = Vec2::new(f32::MAX, f32::MAX);
        let mut new_max = Vec2::new(f32::MIN, f32::MIN);

        for corner in &corners {
            new_min = new_min.min(*corner);
            new_max = new_max.max(*corner);
        }

        (new_min, new_max)
    }

    ///Pushes a translation to the matrix
    pub fn push_translation(&mut self, translation: Vec2) {
        *self = Mat3::from_translation(translation) * *self
    }

    ///Pops a translation from the matrix
    pub fn pop_translation(&mut self) -> Vec2 {
        let translation = self.get_translation();
        *self = Mat3::from_translation(-translation) * *self;
        translation
    }

    ///Creates a billboard matrix that always faces the camera
    pub fn billboard(position: Vec2, size: Vec2) -> Mat3 {
        Mat3::from_translation(position) * Mat3::from_scale(size)
    }

    ///Creates a matrix from origin, scale, and pivot point
    pub fn from_pivot(origin: Vec2, scale: Vec2, pivot: Vec2) -> Mat3 {
        Mat3::from_translation(origin)
            * Mat3::from_translation(pivot)
            * Mat3::from_scale(scale)
            * Mat3::from_translation(-pivot)
    }

    ///Converts to a 4x4 matrix
    pub fn to_mat4(&self) -> Mat4 {
        Mat4::new(
            Vec4::new(self.x.x, self.x.y, 0.0, self.x.z),
            Vec4::new(self.y.x, self.y.y, 0.0, self.y.z),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(self.z.x, self.z.y, 0.0, self.z.z),
        )
    }

    ///Creates a matrix that transforms one rect to another
    pub fn rect_transform(from: (Vec2, Vec2), to: (Vec2, Vec2)) -> Mat3 {
        let from_size = from.1 - from.0;
        let to_size = to.1 - to.0;

        let scale = Vec2::new(to_size.x / from_size.x, to_size.y / from_size.y);

        Mat3::from_translation(to.0) * Mat3::from_scale(scale) * Mat3::from_translation(-from.0)
    }

    ///Creates  amatrix that maintains aspect ratio while fitting a size
    pub fn fit_rect(container: Vec2, content: Vec2) -> Mat3 {
        let scale = (container.x / content.x).min(container.y / content.y);
        let offset = (container - content * scale) * 0.5;

        Mat3::from_translation(offset) * Mat3::from_scale(Vec2::new(scale, scale))
    }

    ///Creates a matrix from translation, rotation, scale, and pivot point
    pub fn from_trs(translation: Vec2, angle: f32, scale: Vec2, pivot: Vec2) -> Mat3 {
        Mat3::from_translation(translation)
            * Mat3::from_translation(pivot)
            * Mat3::from_rotation_z(angle)
            * Mat3::from_scale(scale)
            * Mat3::from_translation(-pivot)
    }

    ///Transforms a rectangle
    pub fn transform_rect(&self, rect: (Vec2, Vec2)) -> [Vec2; 4] {
        let (min, max) = rect;
        [
            self.transform_point(Vec2::new(min.x, min.y)),
            self.transform_point(Vec2::new(min.x, max.y)),
            self.transform_point(Vec2::new(max.x, min.y)),
            self.transform_point(Vec2::new(max.x, max.y)),
        ]
    }

    ///Checks if the matrix contains mirroring/flipping (negative scale)
    pub fn has_mirroring(&self) -> bool {
        self.determinant() < 0.0
    }

    ///Removes mirroring by forcing positive scale
    pub fn remove_mirroring(&self) -> Mat3 {
        let (translation, angle, scale) = self.decompose();
        let positive_scale = Vec2::new(scale.x.abs(), scale.y.abs());
        Mat3::from_translation(translation)
            * Mat3::from_rotation_z(angle)
            * Mat3::from_scale(positive_scale)
    }

    pub fn inverse_or_identity(&self) -> Mat3 {
        self.inverse().unwrap_or_else(|| Mat3::identity())
    }

    ///Returns true if the matrix is all 0.0
    pub fn is_identity(&self) -> bool {
        *self == Mat3::identity()
    }

    ///Returns the matrix as a 2D row-major array
    pub fn to_array_2d_row_major(&self) -> [[f32; 3]; 3] {
        [
            [self.x.x, self.y.x, self.z.x],
            [self.x.y, self.y.y, self.z.y],
            [self.x.z, self.z.y, self.z.z],
        ]
    }

    ///Returns the matrix as a flat row-major array
    pub fn to_array_row_major(&self) -> [f32; 9] {
        [
            self.x.x, self.y.x, self.z.x, self.x.y, self.y.y, self.z.y, self.x.z, self.z.y,
            self.z.z,
        ]
    }

    ///Returns the matrix as a 2D column major array
    pub fn to_array_2d_col_major(&self) -> [[f32; 3]; 3] {
        [
            [self.x.x, self.x.y, self.x.z],
            [self.y.x, self.y.y, self.y.z],
            [self.z.x, self.z.y, self.z.z],
        ]
    }

    ///Returns the matrix as a flat column-major array
    pub fn to_array_col_major(&self) -> [f32; 9] {
        [
            self.x.x, self.x.y, self.x.z, self.y.x, self.y.y, self.y.z, self.z.x, self.z.y,
            self.z.z,
        ]
    }

    ///Returns true if bottom row is [0, 0, 1]
    pub fn is_affine(&self) -> bool {
        epsilon_eq_default(self.z.x, 0.0)
            && epsilon_eq_default(self.z.y, 0.0)
            && epsilon_eq_default(self.z.z, 1.0)
    }

    ///Pretty-prints the matrix for debugging purposes
    pub fn pretty_print(&self) {
        println!("[{:.3} {:.3} {:.3}]", self.x.x, self.y.x, self.z.x);
        println!("[{:.3} {:.3} {:.3}]", self.x.y, self.y.y, self.z.y);
        println!("[{:.3} {:.3} {:.3}]", self.x.z, self.y.z, self.z.z);
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
        Self::new(self * (rhs.x), self * (rhs.y), self * (rhs.z))
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

impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.x.x * v.x + self.y.x * v.y + self.z.x * v.z,
            self.x.y * v.x + self.y.y * v.y + self.z.y * v.z,
            self.x.z * v.x + self.y.z * v.y + self.z.z * v.z,
        )
    }
}

impl Default for Mat3 {
    fn default() -> Self {
        Mat3 {
            x: Vec3::zero(),
            y: Vec3::zero(),
            z: Vec3::zero(),
        }
    }
}

use std::ops::{Index, IndexMut};

impl Index<usize> for Mat3 {
    type Output = Vec3;

    fn index(&self, row: usize) -> &Vec3 {
        match row {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Mat3 row index out of bounds: {}", row),
        }
    }
}

impl IndexMut<usize> for Mat3 {
    fn index_mut(&mut self, row: usize) -> &mut Vec3 {
        match row {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Mat3 row index out of bounds: {}", row),
        }
    }
}
