use crate::{
    epsilon_eq,
    mat3::Mat3,
    mat4::Mat4,
    utils::epsilon_eq_default,
    vec::{Vec3, Vec4},
};

use std::f32::{INFINITY, NAN, consts::PI};
use std::{
    cmp::PartialEq,
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// A quaternion representing a 3D rotation
///
/// Quaternions are stored as `(x, y, z, w)` where `(x, y, z)` is the vector part
/// and `w` is the scalar part. For a unit quaternion representing a rotation,
/// this corresponds to `(axis.sin(angle/2), cos(angle/2))`.
#[derive(Clone, Copy, Debug)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    /// Creates a new quaternion from its raw components.
    #[inline]
    #[must_use]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Quat {
        Quat { x, y, z, w }
    }

    /// An identity quaternion, representing no rotation.
    pub const IDENTITY: Quat = Quat {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    /// A quaternion with all components set to zero.
    pub const ZERO: Quat = Quat {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };

    /// A quaternion with all components set to NaN.
    pub const NAN: Quat = Quat {
        x: NAN,
        y: NAN,
        z: NAN,
        w: NAN,
    };

    /// A quaternion with all components set to INFINITY.
    pub const INFINITY: Quat = Quat {
        x: INFINITY,
        y: INFINITY,
        z: INFINITY,
        w: INFINITY,
    };

    /// Creates a quaternion from a rotation axis and an angle.
    ///
    /// # Parameters
    /// - `axis`: The axis of rotation. Must be a non-zero vector. The vector will be normalized.
    /// - `angle_rad`: The angle of rotation in radians.
    #[must_use]
    pub fn from_axis_angle(axis: Vec3, angle_rad: f32) -> Self {
        let half_angle = angle_rad * 0.5;
        let (s, c) = half_angle.sin_cos();
        let axis = axis.normalize_or_zero();
        Self {
            x: axis.x * s,
            y: axis.y * s,
            z: axis.z * s,
            w: c,
        }
    }

    /// Constructs a quaternion from Euler angles (roll, pitch, yaw) in radians.
    ///
    /// The angles are interpreted as intrinsic Tait-Bryan rotations applied in the order:
    /// **roll (X-axis), then pitch (Y-axis), then yaw (Z-axis)** - also known as XYZ order.
    ///
    /// This follows the right-handed coordinate system convention.
    ///
    /// # Parameters
    /// - `roll`: rotation around the **X-axis**, in radians.
    /// - `pitch`: rotation around the **Y-axis**, in radians.
    /// - `yaw`: rotation around the **Z-axis**, in radians.
    ///
    /// # Returns
    /// A quaternion representing the combined rotation.
    ///
    /// # Example
    /// ```rust
    /// use omelet::quat::Quat;
    /// let q = Quat::from_euler_angles(0.1, 0.2, 0.3);
    /// let (r, p, y) = q.to_euler_angles();
    /// assert!((r - 0.1).abs() < 1e-4);
    /// assert!((p - 0.2).abs() < 1e-4);
    /// assert!((y - 0.3).abs() < 1e-4);
    /// ```
    /// 
    /// # See also
    /// - [`to_euler_angles`] to convert back to Euler angles.
    /// - Consider using quaternions directly when interpolating or avoiding gimbal lock.
    #[must_use]
    pub fn from_euler_angles(roll: f32, pitch: f32, yaw: f32) -> Self {
        let (half_roll_sin, half_roll_cos) = (roll * 0.5).sin_cos(); // X
        let (half_pitch_sin, half_pitch_cos) = (pitch * 0.5).sin_cos(); // Y
        let (half_yaw_sin, half_yaw_cos) = (yaw * 0.5).sin_cos(); // Z

        let x = half_roll_sin * half_pitch_cos * half_yaw_cos
            - half_roll_cos * half_pitch_sin * half_yaw_sin;
        let y = half_roll_cos * half_pitch_sin * half_yaw_cos
            + half_roll_sin * half_pitch_cos * half_yaw_sin;
        let z = half_roll_cos * half_pitch_cos * half_yaw_sin
            - half_roll_sin * half_pitch_sin * half_yaw_cos;
        let w = half_roll_cos * half_pitch_cos * half_yaw_cos
            + half_roll_sin * half_pitch_sin * half_yaw_sin;

        Quat { x, y, z, w }
    }

    /// Creates a quaternion that rotates vector `from` to align with vector `to`.
    ///
    /// # Parameters
    /// - `from`: The starting direction vector. Must be normalized.
    /// - `to`: The ending direction vector. Must be normalized.
    #[must_use]
    pub fn from_to_rotation(from: Vec3, to: Vec3) -> Self {
        let dot = from.dot(to);
        if dot > 0.999999 {
            return Self::IDENTITY;
        } else if dot < -0.999999 {
            let mut axis = Vec3::new(0.0, 1.0, 0.0).cross(from);
            if axis.squared_length() < 1e-6 {
                axis = Vec3::new(1.0, 0.0, 0.0).cross(from);
            }
            return Self::from_axis_angle(axis.normalize(), PI);
        } else {
            let axis = from.cross(to);
            let w = 1.0 + dot;
            Self::new(axis.x, axis.y, axis.z, w).normalize()
        }
    }

    /// Creates a quaternion from a rotation around the X-axis
    #[inline]
    #[must_use]
    pub fn from_rotation_x(angle_rad: f32) -> Self {
        let (s, c) = (angle_rad * 0.5).sin_cos();
        Self {
            x: s,
            y: 0.0,
            z: 0.0,
            w: c,
        }
    }

    /// Creates a quaternion from a rotation around the Y-axis.
    #[inline]
    #[must_use]
    pub fn from_rotation_y(angle_rad: f32) -> Self {
        let (s, c) = (angle_rad * 0.5).sin_cos();
        Self {
            x: 0.0,
            y: s,
            z: 0.0,
            w: c,
        }
    }

    ///Creates a quaternion from a rotation around the Z-axis
    #[inline]
    #[must_use]
    pub fn from_rotation_z(angle_rad: f32) -> Self {
        let (s, c) = (angle_rad * 0.5).sin_cos();
        Self {
            x: 0.0,
            y: 0.0,
            z: s,
            w: c,
        }
    }

    /// Decomposes the rotation into a "swing" and "twist" component.
    /// The twist is the rotation around the `axis`, and the swing is the remaining rotation.
    ///
    /// # Parameters
    /// - `axis`: The axis to twist around. Must be a unit vector.
    ///
    /// # Returns
    /// A tuple `(swing, twist)`. The original rotation can be reconstructed by `swing * twist`.
    #[must_use]
    pub fn to_swing_twist(self, axis: Vec3) -> (Self, Self) {
        let v = self.vector_part();
        let p = axis.dot(v) * axis;
        let twist = Self::new(p.x, p.y, p.z, self.w).normalize();
        let swing = self * twist.conjugate();
        (swing, twist)
    }

    /// Creates a quaternion from the rotational part of a 4x4 matrix.
    ///
    /// This will extract the upper-left 3x3 matrix and convert it to a quaternion.
    /// It assumes the matrix has non-uniform scaling.
    #[must_use]
    pub fn from_mat4(m: &Mat4) -> Self {
        let mat3 = Mat3::new(m.col0.xyz(), m.col1.xyz(), m.col2.xyz());
        Self::from_mat3(&mat3.orthonormalize())
    }

    /// Converts the quaternion to Euler angles (roll, pitch, yaw) in radians.
    ///
    /// The returned angles represent intrinsic rotations about the **X (roll)**,
    /// **Y (pitch)**, and **Z (yaw)** aces, applies in that order (XYZ).
    ///
    /// This corresponds to Tait-Bryan angles using the **right-handed** coordinate system.
    ///
    /// # Returns
    /// A tuple `(roll, pitch, yaw)` of angles in **radians**, where:
    /// - `roll` is rotation around the **X-axis**.
    /// - `pitch` is rotation around the **Y-axis**.
    /// - `yaw` is rotation around the **Z-axis**.
    ///
    /// # Notes
    /// - The conversion may suffer from **gimbal lock** when `pitch` is ±90° (±π/2),
    /// where roll and yaw become coupled.
    /// - The output angles are not guaranteed to exactly match those used to create
    /// the quaternion due to numerical precision and the non-uniqueness of Euler angles.
    ///
    /// # See also
    /// - [`from_euler_angles`] to convert from Euler angles to a quaternion.
    #[must_use]
    pub fn to_euler_angles(self) -> (f32, f32, f32) {
        let Quat { x, y, z, w } = self;

        // roll (x-axis rotation)
        let sinr_cosp = 2.0 * (w * x + y * z);
        let cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
        let roll = sinr_cosp.atan2(cosr_cosp);

        // pitch (y-axis rotation)
        let sinp = 2.0 * (w * y - z * x);
        let pitch = if sinp.abs() >= 1.0 {
            sinp.signum() * std::f32::consts::FRAC_PI_2
        } else {
            sinp.asin()
        };

        // yaw (z-axis rotation)
        let siny_cosp = 2.0 * (w * z + x * y);
        let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        (roll, pitch, yaw)
    }

    /// Calculates the length (magnitude) of the quaternion.
    #[inline]
    #[must_use]
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    /// Calculates the squared length (magnitude) of the quaternion.
    /// This is faster than `length()` as it avoids `sqrt`.
    #[inline]
    #[must_use]
    pub fn squared_length(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    /// Normalizes the quaternion to unit length.
    /// For a zero-length quaternion, this returns `Quat::IDENTITY`.
    #[must_use]
    pub fn normalize(&self) -> Quat {
        let len_sq = self.squared_length();
        if len_sq > 0.0 {
            let inv_len = 1.0 / len_sq.sqrt();
            Self {
                x: self.x * inv_len,
                y: self.y * inv_len,
                z: self.z * inv_len,
                w: self.w * inv_len,
            }
        } else {
            Self::IDENTITY
        }
    }

    /// Normalizes the quaternion to a unit length, returning `None` if the length is zero.
    #[must_use]
    pub fn try_normalize(&self) -> Option<Quat> {
        let len_sq = self.squared_length();
        if len_sq > 0.0 {
            Some(*self * (1.0 / len_sq.sqrt()))
        } else {
            None
        }
    }

    /// Checks if the quaternion is normalized (has a length of 1).
    #[must_use]
    pub fn is_normalized(&self) -> bool {
        epsilon_eq_default(self.squared_length(), 1.0)
    }

    /// Checks if any component of the quaternion is `NaN`.
    #[must_use]
    pub fn is_nan(&self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan() || self.w.is_nan()
    }

    /// Checks if all components of the quaternion are finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }

    /// Computes the conjugate of the quaternion.
    /// For a unit quaternion, the conjugate is the same as the inverse.
    #[inline]
    #[must_use]
    pub fn conjugate(&self) -> Quat {
        Quat {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    /// Computes the inverse of the quaternion.
    /// For a zero-length quaternion, this returns `Quat::IDENTITY`.
    #[must_use]
    pub fn inverse(&self) -> Quat {
        let len_sq = self.squared_length();
        if len_sq > 0.0 {
            self.conjugate() / len_sq
        } else {
            Self::IDENTITY
        }
    }

    /// Computes the dot product of two quaternions.
    #[inline]
    #[must_use]
    pub fn dot(&self, other: Quat) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    /// Rotates a 3D vector by the quaternion.
    ///
    /// Note: Assumes `self` is a unit vector.
    #[must_use]
    pub fn rotate_vec3(&self, v: Vec3) -> Vec3 {
        let q_vec = Vec3::new(self.x, self.y, self.z);
        let t = 2.0 * q_vec.cross(v);
        v + (self.w * t) + q_vec.cross(t)
    }

    /// Calculates the angle in radians between two quaternions.
    #[must_use]
    pub fn angle_between(a: Quat, b: Quat) -> f32 {
        2.0 * a.dot(b).clamp(-1.0, 1.0).acos()
    }

    /// Converts the quaternion to an axis-angle representation.
    ///
    /// # Returns
    /// A tuple `(axis, angle)` where `angle` is in radians.
    /// For an identity quaternion, the axis is `Vec3::X` and the angle is `0.0`.
    #[must_use]
    pub fn to_axis_angle(&self) -> (Vec3, f32) {
        let quat = if self.w > 1.0 {
            self.normalize()
        } else {
            *self
        };

        let angle = 2.0 * quat.w.acos();
        let s = (1.0 - quat.w * quat.w).sqrt();

        if s < 1e-6 {
            (Vec3::new(1.0, 0.0, 0.0), 0.0)
        } else {
            (Vec3::new(quat.x / s, quat.y / s, quat.z / s), angle)
        }
    }

    /// Creates a rotation quaternion that looks along `forward` with the head pointed towards `up`.
    #[must_use]
    pub fn look_rotation(forward: Vec3, up: Vec3) -> Quat {
        let forward = forward.normalize();
        let right = up.cross(forward).normalize();
        let up = forward.cross(right);
        Self::from_mat3(&Mat3::new(right, up, forward))
    }

    /// Converts the quaternion to a 3x3 rotation matrix (column-major).
    ///
    /// Note: Assumes quaternion is normalized.
    #[must_use]
    pub fn to_mat3(&self) -> Mat3 {
        let (x, y, z, w) = (self.x, self.y, self.z, self.w);
        let x2 = x + x;
        let y2 = y + y;
        let z2 = z + z;
        let xx = x * x2;
        let xy = x * y2;
        let xz = x * z2;
        let yy = y * y2;
        let yz = y * z2;
        let zz = z * z2;
        let wx = w * x2;
        let wy = w * y2;
        let wz = w * z2;

        Mat3::new(
            Vec3::new(1.0 - (yy + zz), xy + wz, xz - wy),
            Vec3::new(xy - wz, 1.0 - (xx + zz), yz + wx),
            Vec3::new(xz + wy, yz - wx, 1.0 - (xx + yy)),
        )
    }

    /// Creates a quaternion from a 3x3 rotation matrix.
    #[must_use]
    pub fn from_mat3(m: &Mat3) -> Self {
        let trace = m.col0.x + m.col1.y + m.col2.z;
        if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            Self {
                w: 0.25 * s,
                x: (m.col1.z - m.col2.y) / s,
                y: (m.col2.x - m.col0.z) / s,
                z: (m.col0.y - m.col1.x) / s,
            }
        } else if (m.col0.x > m.col1.y) && (m.col0.x > m.col2.z) {
            let s = (1.0 + m.col0.x - m.col1.y - m.col2.z).sqrt() * 2.0;
            Self {
                w: (m.col1.z - m.col2.y) / s,
                x: 0.25 * s,
                y: (m.col2.x + m.col0.y) / s,
                z: (m.col2.x + m.col0.z) / s,
            }
        } else if m.col1.y > m.col2.z {
            let s = (1.0 + m.col1.y - m.col0.x - m.col2.z).sqrt() * 2.0;
            Self {
                w: (m.col2.x - m.col0.z) / s,
                x: (m.col1.x + m.col0.y) / s,
                y: 0.25 * s,
                z: (m.col2.y + m.col1.z) / s,
            }
        } else {
            let s = (1.0 + m.col2.z - m.col0.x - m.col1.y).sqrt() * 2.0;
            Self {
                w: (m.col0.y - m.col1.x) / s,
                x: (m.col2.x + m.col0.z) / s,
                y: (m.col2.y + m.col2.z) / s,
                z: 0.25 * s,
            }
        }
    }

    /// Creates a `Quat` from a `Vec4`.
    #[inline]
    pub fn from_vec4(v: Vec4) -> Quat {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
            w: v.w,
        }
    }

    /// Converts this `Quat` to a `Vec4`.
    #[inline]
    pub fn to_vec4(&self) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, self.w)
    }

    /// Returns the vector part of the quaternion.
    #[inline]
    pub fn vector_part(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    /// Returns the scalar part of the quaternion.
    #[inline]
    pub fn scalar_part(&self) -> f32 {
        self.w
    }

    /// Checks if `self` is approximately equal to `other` using default epsilon.
    #[must_use]
    pub fn approx_eq(&self, other: &Quat) -> bool {
        epsilon_eq_default(self.x, other.x)
            && epsilon_eq_default(self.y, other.y)
            && epsilon_eq_default(self.z, other.z)
            && epsilon_eq_default(self.w, other.w)
    }

    /// Checks if `self` is approximately equal to `other` using a custom `epsilon`.
    #[must_use]
    pub fn approx_eq_eps(&self, other: &Quat, epsilon: f32) -> bool {
        epsilon_eq(self.x, other.x, epsilon)
            && epsilon_eq(self.y, other.y, epsilon)
            && epsilon_eq(self.z, other.z, epsilon)
            && epsilon_eq(self.w, other.w, epsilon)
    }

    /// Linear interpolation between `self` and `b`.
    /// Not guaranteed to produce a unit quaternion.
    #[must_use]
    pub fn lerp(&self, b: Quat, t: f32) -> Quat {
        *self * (1.0 - t) + b * t
    }

    /// Linear interpolation between `a` and `b`.
    /// Not guaranteed to produce a unit quaternion.
    #[must_use]
    pub fn lerp_between(a: Quat, b: Quat, t: f32) -> Quat {
        a * (1.0 - t) + b * t
    }

    /// Normalized linear interpolation between `self` and `b`.
    /// Faster than `slerp` but may not produce as uniform of a rotation speed.
    #[must_use]
    pub fn nlerp(&self, b: Quat, t: f32) -> Quat {
        (*self * (1.0 - t) + b * t).normalize()
    }

    /// Normalized linear interpolation between `a` and `b`.
    /// Faster than `slerp` but may not produce as uniform of a rotation speed.
    #[must_use]
    pub fn nlerp_between(a: Quat, b: Quat, t: f32) -> Quat {
        (a * (1.0 - t) + b * t).normalize()
    }

    /// Spherical linear interpolation between two quaternions.
    /// Produces a unit quaternion with uniform rotation speed.
    #[must_use]
    pub fn slerp(self, end: Self, t: f32) -> Self {
        let mut dot = self.dot(end);
        let mut end_adj = end;

        if dot < 0.0 {
            dot = -dot;
            end_adj = -end;
        }

        if dot > 0.9995 {
            return self.lerp(end_adj, t).normalize();
        }

        let theta_0 = dot.acos();
        let theta = theta_0 * t;
        let sin_theta = theta.sin();
        let sin_theta_0 = theta_0.sin();

        let s0 = (theta_0 - theta).sin() / sin_theta_0;
        let s1 = sin_theta / sin_theta_0;

        (self * s0) + (end_adj * s1)
    }

    /// Computes `e^(self)`.
    pub fn exp(self) -> Quat {
        let v = Vec3::new(self.x, self.y, self.z);
        let len = v.length();
        let (s, c) = len.sin_cos();

        if len < 1e-6 {
            Self {
                x: v.x,
                y: v.y,
                z: v.z,
                w: c,
            }
        } else {
            let scale = s / len;
            Self {
                x: v.x * scale,
                y: v.y * scale,
                z: v.z * scale,
                w: c,
            }
        }
    }

    /// Computes the natural logarithm of a quaternion.
    pub fn log(self) -> Quat {
        let v = Vec3::new(self.x, self.y, self.z);
        let v_len_sq = v.squared_length();
        let q_len_sq = self.squared_length();

        if v_len_sq < 1e-12 {
            Self {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: q_len_sq.sqrt().ln(),
            }
        } else {
            let v_len = v_len_sq.sqrt();
            let q_len = q_len_sq.sqrt();
            let scale = v_len.atan2(self.w) / v_len;
            Self {
                x: v.x * scale,
                y: v.y * scale,
                z: v.z * scale,
                w: q_len.ln(),
            }
        }
    }

    /// Raises the quaternion to a floating-point power.
    pub fn powf(self, exponent: f32) -> Quat {
        (self.log() * exponent).exp()
    }
}

// ============= Operator Overloads =============

/// Adds two quaternions together component-wise.
impl Add for Quat {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
            self.w + rhs.w,
        )
    }
}

/// Subtracts `rhs` from `self` component-wise.
impl Sub for Quat {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z,
            self.w - rhs.w,
        )
    }
}

/// Multiplies two quaternions. This combines their rotations.
impl Mul for Quat {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }
}

/// Rotates a `Vec3` by the quaternion.
impl Mul<Vec3> for Quat {
    type Output = Vec3;
    #[inline]
    fn mul(self, rhs: Vec3) -> Vec3 {
        self.rotate_vec3(rhs)
    }
}

/// Multiplies each component of the quaternion by a scalar.
impl Mul<f32> for Quat {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
            w: self.w * rhs,
        }
    }
}

/// Divides each component of the quaternion by a scalar.
impl Div<f32> for Quat {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f32) -> Self {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
            w: self.w / rhs,
        }
    }
}

/// Negates each component of the quaternion.
impl Neg for Quat {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}

// ============= Assignment Operator Overloads =============

impl AddAssign for Quat {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
        self.w += rhs.w;
    }
}

impl SubAssign for Quat {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
        self.w -= rhs.w;
    }
}

impl MulAssign for Quat {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<f32> for Quat {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
        self.w *= rhs;
    }
}

impl DivAssign<f32> for Quat {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
        self.w /= rhs;
    }
}

// ============= Trait Implementations =============

impl Default for Quat {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Checks whether two quaternions are exactly equal.
impl PartialEq for Quat {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z && self.w == other.w
    }
}

/// Enables `q[index]` access. Panics if `index` is out of bounds.
impl Index<usize> for Quat {
    type Output = f32;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Quat index out of bounds: {}", index),
        }
    }
}

/// Enables mutable `q[index]` access. Panics if `index` is out of bounds.
impl IndexMut<usize> for Quat {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Quat index out of bounds: {}", index),
        }
    }
}

/// Implements the `Display` trait for pretty-printing.
impl fmt::Display for Quat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Quat({:.3}, {:.3}, {:.3}, {:.3})",
            self.x, self.y, self.z, self.w
        )
    }
}

// ============= Conversion Traits =============

/// Creates a `Quat` from a `Vec4`.
impl From<Vec4> for Quat {
    #[inline]
    fn from(v: Vec4) -> Self {
        Self::new(v.x, v.y, v.z, v.w)
    }
}

/// Creates a `Vec4` from a `Quat`.
impl From<Quat> for Vec4 {
    #[inline]
    fn from(q: Quat) -> Self {
        Self::new(q.x, q.y, q.z, q.w)
    }
}

// ============= Approx Crate Implementations =============

/// Implements absolute difference equality comparison for `Quat`.
impl approx::AbsDiffEq for Quat {
    type Epsilon = f32;

    #[inline]
    fn default_epsilon() -> f32 {
        f32::EPSILON
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: f32) -> bool {
        f32::abs_diff_eq(&self.x, &other.x, epsilon)
            && f32::abs_diff_eq(&self.y, &other.y, epsilon)
            && f32::abs_diff_eq(&self.z, &other.z, epsilon)
            && f32::abs_diff_eq(&self.w, &other.w, epsilon)
    }
}

/// Implements relative equality comparison for `Quat`.
impl approx::RelativeEq for Quat {
    #[inline]
    fn default_max_relative() -> f32 {
        f32::EPSILON
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: f32, max_relative: f32) -> bool {
        f32::relative_eq(&self.x, &other.x, epsilon, max_relative)
            && f32::relative_eq(&self.y, &other.y, epsilon, max_relative)
            && f32::relative_eq(&self.z, &other.z, epsilon, max_relative)
            && f32::relative_eq(&self.w, &other.w, epsilon, max_relative)
    }
}
