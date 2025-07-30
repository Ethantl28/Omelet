use rand::Rng;

use crate::mat2::Mat2;
use crate::utils;
use crate::utils::epsilon_eq;
use crate::utils::epsilon_eq_default;
use crate::utils::is_near_zero_default;
use crate::vec::Vec3;
use core::f32;
use std::f32::INFINITY;
use std::f32::NAN;
use std::fmt;

use std::cmp::PartialEq;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// A 2D vector with `x` and `y` components.
///
/// The vector is represented by two `f32` values.
#[derive(Debug, Clone, Copy)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    // ============= Construction and Conversion =============

    /// Creates a new 2D vector with the given `x` and `y` components.
    ///
    /// # Parameters
    /// - `x`: The x component of the vector.
    /// - `y`: The y component of the vector.
    ///
    /// # Returns
    /// A new `Vec2` instance with the specified components.
    pub const fn new(x: f32, y: f32) -> Vec2 {
        Vec2 { x, y }
    }

    /// Converts the vector into an array of two `f32` components `[x, y]`.
    ///
    /// # Returns
    /// An array `[f32; 2]` representing the vector components.
    pub fn to_array(self) -> [f32; 2] {
        [self.x, self.y]
    }

    /// Creates a `Vec2` from an array containing two `f32` values.
    ///
    /// # Parameters
    /// - `array`: A `[f32; 2]` array representing the `x` and `y` components.
    ///
    /// # Returns
    /// A new `Vec2` with components initialized from the array.
    pub fn from_array(array: [f32; 2]) -> Vec2 {
        Vec2::new(array[0], array[1])
    }

    pub fn to_tuple(self) -> (f32, f32) {
        (self.x, self.y)
    }

    pub fn from_tuple(t: (f32, f32)) -> Vec2 {
        Vec2::new(t.0, t.1)
    }

    // ============= Constants ==============
    /// Returns a `Vec2` where all components are zero.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// Returns a `Vec2` where all components are NaN.
    pub const NAN: Self = Self { x: NAN, y: NAN };

    /// Returns a `Vec2` where all components are infinity.
    pub const INFINITY: Self = Self {
        x: INFINITY,
        y: INFINITY,
    };

    /// Returns a `Vec2` equivalent to `(1.0, 0.0)`.
    pub const X: Self = Self { x: 1.0, y: 0.0 };

    /// Returns a `Vec2` equivalent to `(0.0, 1.0)`.
    pub const Y: Self = Self { x: 0.0, y: 1.0 };

    // ============= Math Utilities =============

    /// Returns a new vector with each component replaced by its absolute value.
    ///
    /// This is useful for obtaining the magnitude of each axis regardless of sign.
    ///
    /// # Returns
    /// A `Vec2` where `x` and `y` are the absolute values of the original vector's components.
    pub fn abs(self) -> Vec2 {
        Vec2::new(self.x.abs(), self.y.abs())
    }

    /// Returns a new vector where each component is replaced by its sign.
    ///
    /// The sign of each component can be `-1.0`, `0.0`, or `1.0` depending on whether the
    /// component is negative, zero, or positive respectively.
    ///
    /// # Returns
    /// A `Vec2` representing the sign of each component.

    pub fn signum(self) -> Self {
        Vec2::new(
            if self.x > 0.0 {
                1.0
            } else if self.x < 0.0 {
                -1.0
            } else {
                0.0
            },
            if self.y > 0.0 {
                1.0
            } else if self.y < 0.0 {
                -1.0
            } else {
                0.0
            },
        )
    }

    /// Returns a new vector where each component is replaced by its IEEE 754 signum.
    ///
    /// Unlike [`signum()`](#method.signum), this method treats zero as positive:
    /// `0.0.signum()` is `1.0`, and `-0.0.signum()` is `-1.0`.
    ///
    /// # Returns
    /// A `Vec2` where each component is `-1.0` if negative, `1.0` otherwise.

    pub fn ieee_signum(self) -> Self {
        Vec2::new(self.x.signum(), self.y.signum())
    }

    /// Returns a `Vec3` where `z` is the Z-axis of the `Vec3`.
    pub fn extend(&self, z: f32) -> Vec3 {
        Vec3::new(self.x, self.y, z)
    }

    /// Clamps each component of the vector between the specified `min` and `max` values.
    ///
    /// # Parameters
    /// - `min`: The minimum allowed value for each component.
    /// - `max`: the maximum allowed value for each component.
    ///
    /// # Returns
    /// A new `Vec2` where each component is limited to the range `[min, max]`.
    pub fn clamp(self, min: f32, max: f32) -> Vec2 {
        Vec2::new(
            utils::clamp(self.x, min, max),
            utils::clamp(self.y, min, max),
        )
    }

    /// Returns the component-wise minimum between `self` and another vector.
    ///
    /// # Parameters
    /// - `other`: The other vector.
    ///
    /// # Returns
    /// A `Vec2` with each component being the minimum of the corresponding components.
    ///
    /// # Example
    /// ```rust
    /// use omelet::vec2::Vec2;
    /// let a = Vec2::new(1.0, 4.0);
    /// let b = Vec2::new(3.0, 1.0);
    /// assert_eq!(a.min(b), Vec2::new(1.0, 1.0));
    /// ```
    pub fn min(self, other: Vec2) -> Vec2 {
        Vec2::new(self.x.min(other.x), self.y.min(other.y))
    }

    /// Returns the component-wise maximum between `self` and another vector.
    ///
    /// # Parameters
    /// - `other`: The other vector.
    ///
    /// # Returns
    /// A `Vec2` with each component being the maximum of the corresponding components.
    ///
    /// # Example
    /// ```rust
    /// use omelet::vec2::Vec2;
    /// let a = Vec2::new(1.0, 4.0);
    /// let b = Vec2::new(3.0, 2.0);
    /// assert_eq!(a.max(b), Vec2::new(3.0, 4.0));
    /// ```
    pub fn max(self, other: Vec2) -> Vec2 {
        Vec2::new(self.x.max(other.x), self.y.max(other.y))
    }

    // ============= Magnitude and Normalization =============

    /// Calculates and returns the magnitude (length) of the vector.
    ///
    /// The length is computed as the squre root of the sum of sqaures of `x` and `y`.
    ///
    /// # Returns
    /// A `f32` representing the Euclidean length of the vector.
    ///
    /// Note: This calculation uses sqrt(), which can be taxing on the system.
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Computes and returns the squared magnitude (length) of the vector.
    ///
    /// This avoids the expensive square root calculation used in `length()`,
    /// and it useful when only relative length or comparisons are needed.
    ///
    /// # Returns
    /// A `f32` representing the squared length `(x² + y²)`.
    pub fn squared_length(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    /// Returns a normalized (unit length) vector in the same direction as `self`.
    ///
    /// # Panics
    /// This function panics if the vector length is zero to prevent division by zero.
    ///
    /// # Returns
    /// A `Vec2` with length 1 pointing in the same direction.
    pub fn normalize(&self) -> Vec2 {
        let len = self.length();
        assert!(len != 0.0, "Cannot normalize zero-length vector");
        Vec2::new(self.x / len, self.y / len)
    }

    /// Attempts to return a normalized version of the vector, or `None` if the length
    /// is zero or nearly zero.
    ///
    /// # Returns
    /// - `Some(Vec2)` containing the normalized vector if length > epsilon.
    /// - `None` if vector length is zero or near zero (within `1e-6`).
    pub fn try_normalize(&self) -> Option<Vec2> {
        let len = self.length();
        if is_near_zero_default(len) {
            None
        } else {
            Some(Vec2::new(self.x / len, self.y / len))
        }
    }

    /// Returns a normalized (unit length) vector in the same direction as `self`.
    ///
    /// If the length is near 0 within the default tolerance, then it will return a zero vector.
    ///
    /// # Returns
    /// `Vec2` either normalized or zero.
    pub fn normalize_or_zero(&self) -> Vec2 {
        let len = self.length();
        if epsilon_eq_default(len, 0.0) {
            Self::ZERO
        } else {
            *self / len
        }
    }

    /// Checks whether both components of the vector are approx zero.
    ///
    /// # Returns
    /// `true` if both `x` and `y` are approx equal to 0.0.
    pub fn is_zero(&self) -> bool {
        epsilon_eq_default(self.x, 0.0) && epsilon_eq_default(self.y, 0.0)
    }

    /// Checks whether both components of the vector are approx equal to zero within a user-specified tolerance.
    ///
    /// Useful to check for effective zero while avoiding floating point precision issues.
    ///
    /// # Parameters
    /// - `epsilon`: The allowed margin of error.
    ///
    /// # Returns
    /// `true` if both `x` and `y` are approx equal to zero within epsilon, `false` otherwise.
    pub fn is_zero_eps(&self, epsilon: f32) -> bool {
        epsilon_eq(self.x, 0.0, epsilon) && epsilon_eq(self.y, 0.0, epsilon)
    }

    /// Returns `true` if the vector is normalized within the small epsilon tolerance.
    ///
    /// This checks if the length is approx 1.0 within 1e-6.
    ///
    /// # Returns
    /// Boolean indicating normalization status.
    pub fn is_normalized(self) -> bool {
        epsilon_eq(self.length(), 1.0, 1e-6)
    }

    /// Faster approx check if the squared length is approx 1 within epsilon.
    ///
    /// Avoids computing a square root, useful for performance-critical code.
    ///
    /// # Returns
    /// Boolean indicating if vector is approx normalized.
    pub fn is_normalized_fast(self) -> bool {
        epsilon_eq(self.squared_length(), 1.0, 1e-6)
    }

    // ============= Dot, Cross, and Angles =============

    /// Computes the dot product between `self` and another vector.
    ///
    /// The dor product is defined as `x1 * x2 + y1*y2` and measures the similarity
    /// of the two vectors directions.
    ///
    /// # Parameters
    /// - `other`: The other `Vec2` to perfom the dot with.
    ///
    /// # Returns
    /// A scalar `f32` value representing the dot product.
    pub fn dot(&self, other: Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// Computes the scalar 2D cross product (z-component) of `self` and another vector.
    ///
    /// This is defined as `x1*y2 - y1*x2` and represents the signed magnitude of the
    /// vector perpendicular to the plane containing `self` and `other`.
    ///
    /// # Parameters
    /// - `other`: The other `Vec2` to perform the cross product with.
    ///
    /// # Returns
    /// A scalar `f32` representing the cross product magnitude.
    pub fn cross(&self, other: Vec2) -> f32 {
        self.x * other.y - self.y * other.x
    }

    /// Calculates the angle (in radians) of a 2D vector relative to the x-axis.
    ///
    /// # Returns
    /// `f32` that is the result of the calculation `atan2(self.y, self.x)`, converted to radians.
    pub fn angle_radians(&self) -> f32 {
        utils::degrees_to_radians(self.y.atan2(self.x))
    }

    /// Calculates the angle (in degrees) of a 2D vector relative to the x-axis.
    ///
    /// # Returns
    /// `f32` that is the result of the calculation `atan2(self.y, self.x)`.
    pub fn angle_degrees(&self) -> f32 {
        self.y.atan2(self.x)
    }

    /// Calculates the angle in radians between `self` and another vector.
    ///
    /// The angle is in the range `[0, π]`.
    ///
    /// # Parameters
    /// - `other`: The other vector.
    ///
    /// # Returns
    /// The angle `f32` between the two vectors in radans.
    pub fn angle_between_radians(&self, other: Vec2) -> f32 {
        let theta = (self.dot(other)) / (self.length() * other.length());
        utils::clamp(theta, -1.0, 1.0).acos()
    }

    /// Calculates the angle in degrees between `self` and another vector.
    ///
    /// The angle is in the range `[0°, 180°]`.
    ///
    /// # Parameters
    /// - `other`: The other vector.
    ///
    /// # Returns
    /// The angle `f32` between vectors in degrees.
    pub fn angle_between_degrees(&self, other: Vec2) -> f32 {
        let theta = (self.dot(other)) / (self.length() * other.length());
        utils::radians_to_degrees(utils::clamp(theta, -1.0, 1.0).acos())
    }

    /// Returns the signed angle to another vector in radians.
    ///
    /// # Parameters
    /// - `other`: Vector to find angle to.
    ///
    /// # Returns
    /// `f32` angle from `self` to `other` in radians.
    pub fn angle_to_radians(&self, other: Vec2) -> f32 {
        (self.x * other.y - self.y * other.x).atan2(self.dot(other))
    }

    /// Returns the signed angle to another vector in degrees.
    ///
    /// # Parameters
    /// - `other`: Vector to find angle to.
    ///
    /// # Returns
    /// `f32` angle from `self` to `other` in degrees.
    pub fn angle_to_degrees(&self, other: Vec2) -> f32 {
        utils::radians_to_degrees((self.x * other.y - self.y * other.x).atan2(self.dot(other)))
    }

    /// Creates a new unit vector from an angle (degrees).
    ///
    /// # Parameters
    /// - `angle`: `f32` to create the vector from.
    ///
    /// # Returns
    /// `Vec2` that was created using this formula: `(angle.cos(), angle.sin())`.
    pub fn from_angle(angle: f32) -> Vec2 {
        Vec2::new(angle.cos(), angle.sin())
    }

    // ============= Interpolation =============

    /// Linearly interpolates between `self` and vector `b` by factor `t`.
    ///
    /// This performs a weighted average:
    /// `result = self * (1.0 - t) + b * t`
    ///
    /// # Parameters
    /// - `b`: The target vector to interpolate towards.
    /// - `t`: The interpolation factor in the range `[0.0, 1.0]`.
    ///
    /// # Returns
    /// A `Vec2` representing the point `t` fraction between `self` and `b`.
    pub fn lerp(&self, b: Vec2, t: f32) -> Vec2 {
        *self * (1.0 - t) + b * t
    }

    /// Clamped version of `lerp` that restricts `t` to the range `[0.0, 1.0]`.
    ///
    /// Prevents overshooting during interpolation.
    ///
    /// # Parameters
    /// - `b`: The target vector.
    /// - `t`: The interpolation factor, automatically clamped between `0.0` and `1.0`.
    ///
    /// # Returns
    /// A clamped interpolated `Vec2`.
    pub fn lerp_clamped(&self, b: Vec2, t: f32) -> Vec2 {
        let t = utils::clamp(t, 0.0, 1.0);
        *self * (1.0 - t) + b * t
    }

    /// Linearly interpolates between two vectors `a` and `b` by factor `t`.
    ///
    /// Unlike `lerp` and `lerp_clamped`, this is a static method that does not require `self`.
    ///
    /// # Parameters
    /// - `a`: The starting vector.
    /// - `b`: The target vector.
    /// - `t`: The interpolation factor.
    ///
    /// # Returns
    /// A vector interpolated between `a` and `b`.
    #[inline]
    pub fn lerp_between(a: Vec2, b: Vec2, t: f32) -> Vec2 {
        a * (1.0 - t) + b * t
    }

    /// Same as `lerp_between`, but clamps `t` to `[0.0, 1.0]`.
    ///
    /// Ensures the result remains between `a` and `b`.
    ///
    /// # Parameters
    /// - `a`: The starting vector.
    /// - `b`: The target vector.
    /// - `t`: The interpolation factor, clamped.
    ///
    /// # Returns
    /// A clamped vector interpolated between `a` and `b`.
    #[inline]
    pub fn lerp_between_clamped(a: Vec2, b: Vec2, t: f32) -> Vec2 {
        let t = utils::clamp(t, 0.0, 1.0);
        a * (1.0 - t) + b * t
    }

    /// Performs spherical linear interpolation (slerp) between two 2D vectors.
    ///
    /// This version blends both angle and magnitude, yielding a curved interpolation path.
    /// Handled edge cases for nearly-parallel, nearly-opposite, or zero vectors.
    ///
    /// # Parameters
    /// - `a`: Start vector.
    /// - `b`: End vector.
    /// - `t`: Interpolation parameter between 0 and 1.
    ///
    /// # Returns
    /// A vector interpolated on the shortest arc between `a` and `b`.
    ///
    /// /// # Example
    /// ```rust
    /// use omelet::vec2::Vec2;
    /// let a = Vec2::new(1.0, 0.0);
    /// let b = Vec2::new(0.0, 1.0);
    /// let halfway = Vec2::slerp(a, b, 0.5);
    /// assert!((halfway.length() - 1.0).abs() < 1e-6);
    /// ```
    pub fn slerp(a: Vec2, b: Vec2, t: f32) -> Vec2 {
        // Handle zero vectors
        if a.length() < 1e-6 {
            return b * t;
        }
        if b.length() < 1e-6 {
            return a * (1.0 - t);
        }

        let a_norm = a.normalize();
        let b_norm = b.normalize();
        let dot = a_norm.dot(b_norm).clamp(-1.0, 1.0);

        // Handle nearly parallel vectors
        if dot > 1.0 - 1e-6 {
            return Vec2::lerp_between(a, b, t);
        }

        // Handle opposite directions
        if dot < -1.0 + 1e-6 {
            // Find perpendicular vector
            let perp = Vec2::new(-a_norm.y, a_norm.x);
            // Interpolate length
            let length = a.length() * (1.0 - t) + b.length() * t;
            // Rotate through perpendicular
            return if t < 0.5 {
                perp.normalize() * length
            } else {
                (-perp).normalize() * length
            };
        }

        let theta = dot.acos();
        let sin_theta = theta.sin();
        let wa = ((1.0 - t) * theta).sin() / sin_theta;
        let wb = (t * theta).sin() / sin_theta;

        // Interpolate both direction and magnitude
        (a_norm * wa + b_norm * wb) * (a.length() * (1.0 - t) + b.length() * t)
    }

    /// A simplified 2D spherical interpolation using angle-based interpolation.
    ///
    /// Instead of blending vector directions directly, this version interpolates the
    /// angle between `a` and `b` and reconstructs the vector from it.
    ///
    /// # Parameters
    /// - `a`: First vector.
    /// - `b`: Second vector.
    /// - `t`: Interpolation factor.
    ///
    /// # Returns
    /// A vector pointing in the interpolated direction between `a` and `b`, scaled by interpolated magnitude.
    ///
    /// /// # Example
    /// ```rust
    /// use omelet::vec2::Vec2;
    /// let a = Vec2::new(1.0, 0.0);
    /// let b = Vec2::new(0.0, 1.0);
    /// let result = Vec2::slerp_angle(a, b, 0.5);
    /// assert!((result.length() - 1.0).abs() < 1e-6);
    /// ```
    pub fn slerp_angle(a: Vec2, b: Vec2, t: f32) -> Vec2 {
        // Handle zero vectors
        if a.length() < 1e-6 || b.length() < 1e-6 {
            return Vec2::lerp(&a, b, t);
        }

        let a_norm = a.normalize();
        let b_norm = b.normalize();

        // Calculate angles
        let angle_a = a_norm.y.atan2(a_norm.x);
        let angle_b = b_norm.y.atan2(b_norm.x);

        // Find the shortest angular distance
        let mut delta_angle = (angle_b - angle_a).rem_euclid(std::f32::consts::PI * 2.0);
        if delta_angle > std::f32::consts::PI {
            delta_angle -= std::f32::consts::PI * 2.0;
        }

        // Interpolate angle and length separately
        let angle = angle_a + t * delta_angle;
        let length = a.length() * (1.0 - t) + b.length() * t;

        Vec2::new(angle.cos(), angle.sin()) * length
    }

    // ============= Projection and Reflection =============

    /// Project `self` onto another vector `onto`.
    ///
    /// This returns the vector component of `self` that lies in the direction of `onto`.
    /// If `onto` is a zero vector, returns the zero vector to avoid division by zero.
    ///
    /// # Parameters
    /// - `onto`: The vector to project onto.
    ///
    /// # Returns
    /// The projection vector of `self` onto `onto`.
    ///
    /// # Example
    /// ```rust
    /// use omelet::vec2::Vec2;
    /// let v = Vec2::new(3.0, 4.0);
    /// let onto = Vec2::new(1.0, 0.0);
    /// assert_eq!(v.project(onto), Vec2::new(3.0, 0.0));
    /// ```
    pub fn project(&self, onto: Vec2) -> Vec2 {
        let denominator = onto.dot(onto);
        if denominator == 0.0 {
            Vec2::new(0.0, 0.0)
        } else {
            onto * (self.dot(onto) / denominator)
        }
    }

    /// Returns the component of `self` orthogonal to the vector `other`.
    ///
    /// The rejection is defined as `self - projection of self onto other`.
    ///
    /// # Parameters
    /// - `other`: The vector to reject from.
    ///
    /// # Returns
    /// A `Vec2` representing the component of `self` perpendicular to `other`.
    ///
    /// # Example
    /// ```rust
    /// use omelet::vec2::Vec2;
    /// let v = Vec2::new(3.0, 4.0);
    /// let onto = Vec2::new(1.0, 0.0);
    /// assert_eq!(v.reject(onto), Vec2::new(0.0, 4.0));
    /// ```
    pub fn reject(&self, other: Vec2) -> Vec2 {
        *self - self.project(other)
    }

    /// Reflects the vector `self` over a given normal vector.
    ///
    /// This performs a mirror reflection of the vector abou the plane defined by the normal.
    /// The normal vector is assumed to be normalized.
    ///
    /// # Parameters
    /// - `normal`: The normalized normal vector to reflect about.
    ///
    /// # Returns
    /// A `Vec2` reflected.
    ///
    /// # Example
    /// ```rust
    /// use omelet::vec2::Vec2;
    /// let v = Vec2::new(1.0, -1.0);
    /// let normal = Vec2::new(0.0, 1.0);
    /// let reflected = v.reflect(normal);
    /// assert_eq!(reflected, Vec2::new(1.0, 1.0));
    /// ```
    pub fn reflect(&self, normal: Vec2) -> Vec2 {
        *self - normal * (2.0 * self.dot(normal))
    }

    /// Returns the mirrored vector over a given normal.
    ///
    /// # Parameters
    /// - `normal`: The normal to mirror the vector over.
    ///
    /// # Returns
    /// `Vec2` which is mirrored over `normal`.
    pub fn mirror(&self, normal: Vec2) -> Vec2 {
        *self - normal * (2.0 * self.dot(normal))
    }

    // ============= Distance =============

    /// Returns the Euclidean distance between `self` and another vector.
    ///
    /// Calculated as the length of the difference vector.
    ///
    /// # Parameters
    /// - `other`: The target vector.
    ///
    /// # Returns
    /// A `f32` distance value.
    pub fn distance(&self, other: Vec2) -> f32 {
        (*self - other).length()
    }

    /// Returns the squared Euclidean distance between `self` and another vector.
    ///
    /// This avoids the cost of the square root and is useful when only relative comparisons
    /// of distance are required.
    ///
    /// # Parameters
    /// - `other`: The target vector.
    ///
    /// # Returns
    /// A `f32` squared distance value.
    pub fn squared_distance(&self, other: Vec2) -> f32 {
        (*self - other).squared_length()
    }

    /// Returns the normalized direction vector pointing from `self` towards `other`.
    ///
    /// # Parameters
    /// - `other`: The target vector.
    ///
    /// # Returns
    /// A unit length `Vec2` pointing from `self` to `other`.
    ///
    /// # Panics
    /// Panics if the direction vector length is zero (i.e. both vectors equal).
    pub fn direction_to(&self, other: Vec2) -> Vec2 {
        let delta = other - *self;
        delta.normalize()
    }

    /// Returns the raw direction vector from `self` to `other` without normalization.
    ///
    /// # Parameters
    /// - `other`: The target vector.
    ///
    /// # Returns
    /// The difference vector `other - self`.
    pub fn direction_to_raw(&self, other: Vec2) -> Vec2 {
        other - *self
    }

    // ============= Geometry =============

    /// Returns a vector perpendicular to `self`, rotated 90 degrees counter-clockwise.
    ///
    /// This is useful for generating vectors in 2D.
    ///
    /// # Returns
    /// A `Vec2` perpendicular to `self`.
    pub fn perpendicular(self) -> Vec2 {
        Vec2::new(-self.y, self.x)
    }

    /// Returns a vector perpendicular to `self` rotates 90 degrees clockwise.
    ///
    /// # Returns
    /// `Vec2` perpendicular to `self`.
    pub fn normal(&self) -> Vec2 {
        Vec2::new(self.y, -self.x)
    }

    /// Moves the `current` vector towards the `target` vector by at most `max_delta`.
    ///
    /// If the distance between `current` and `target` is less than or equal to `max_delta`,
    /// returns `target`. Otherwise, returns a vector closer to `target` by `max_delta`.
    ///
    /// # Parameters
    /// - `current`: The starting position vector.
    /// - `target`: The target position vector.
    /// - `max_delta`: The maximum distance to move towards the target.
    ///
    /// # Returns
    /// A new `Vec2` moved towards the target by `max_delta` or exactly `target` if close enough.
    ///
    /// # Example
    /// ```rust
    /// use omelet::vec2::Vec2;
    /// let current = Vec2::new(0.0, 0.0);
    /// let target = Vec2::new(10.0, 0.0);
    /// let moved = Vec2::move_towards(current, target, 3.0);
    /// assert_eq!(moved, Vec2::new(3.0, 0.0));
    /// ```
    pub fn move_towards(current: Vec2, target: Vec2, max_delta: f32) -> Vec2 {
        let delta = target - current;
        let distance = delta.length();

        if distance <= max_delta || distance < f32::EPSILON {
            target
        } else {
            current + delta / distance * max_delta
        }
    }

    /// Rotates the vector by the given angle in radians (counter-clockwise).
    ///
    /// This uses a 2x2 rotation matrix internally.
    ///
    /// # Parameters
    /// - `angle_rad`: The angle in radians to rotate by.
    pub fn rotate(&mut self, angle_rad: f32) {
        *self = Mat2::from_angle(angle_rad) * *self;
    }

    /// Rotates the vector around a given point by `angle` radians.
    ///
    /// # Parameters
    /// - `center`: `Vec2` which represents the center for the rotation.
    /// - `angle`: `f32` which represents the angle to rotate the vector by (in radians).
    ///
    /// # Returns
    /// `Vec2` rotated by `angle` radians around `center`.
    pub fn rotate_around(&self, center: Vec2, angle: f32) -> Vec2 {
        let sin = angle.sin();
        let cos = angle.cos();
        let translated = *self - center;
        let rotated = Vec2::new(
            translated.x * cos - translated.y * sin,
            translated.x * sin + translated.y * cos,
        );
        rotated + center
    }

    // ============= Random =============

    /// Returns a randomly generated unit vector using rand crate.
    ///
    /// # Returns
    /// `Vec2` which each component is randomly generated in range `[0.0, 1.0]`.
    pub fn random_unit_vector() -> Vec2 {
        let mut rng = rand::rng();
        loop {
            let x = rng.random_range(-1.0..=1.0);
            let y = rng.random_range(-1.0..=1.0);

            let v = Vec2::new(x, y);
            let len_sq = v.squared_length();
            if len_sq > 0.0 && len_sq <= 1.0 {
                return v / len_sq.sqrt();
            }
        }
    }

    // ============= Barycentric and Triangles =============

    /// Computes the barycentric coordinates of point `p` relative to triangle (`a`, `b`, `c`).
    ///
    /// Useful for interpolation inside triangles or determining whether a point lies in a triangle.
    ///
    /// # Parameters
    /// - `p`: The point to evaluate.
    /// - `a`, `b`, `c`: The triangles vertices.
    ///
    /// # Returns
    /// A tupe `(u, v, w)` of barycentric weight where `u + v + w = 1`.
    ///
    /// /// # Example
    /// ```rust
    /// use omelet::vec2::Vec2;
    /// let p = Vec2::new(2.0, 1.0);
    /// let a = Vec2::new(0.0, 0.0);
    /// let b = Vec2::new(4.0, 0.0);
    /// let c = Vec2::new(0.0, 4.0);
    /// let (u, v, w) = Vec2::barycentric(p, a, b, c);
    /// assert!((u + v + w - 1.0).abs() < 1e-6);
    /// ```
    pub fn barycentric(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> (f32, f32, f32) {
        let v0 = b - a;
        let v1 = c - a;
        let v2 = p - a;

        let d00 = v0.dot(v0);
        let d01 = v0.dot(v1);
        let d11 = v1.dot(v1);
        let d20 = v2.dot(v0);
        let d21 = v2.dot(v1);

        let denom = d00 * d11 - d01 * d01;
        let v = (d11 * d20 - d01 * d21) / denom;
        let w = (d00 * d21 - d01 * d20) / denom;
        let u = 1.0 - v - w;

        (u, v, w)
    }

    /// Interpolated a point using barycentric weights.
    ///
    /// Different to `barycentric` as this does not do in depth coordinate calculations.
    ///
    /// # Parameters
    /// - `a`: First vector.
    /// - `b`: Second vector.
    /// - `c`: Third vector.
    /// - `u`: First weight.
    /// - `v`: Second weight.
    /// - `w`: Third weight.
    ///
    /// # Returns
    /// A `Vec2` interpolated using barycentric weights.
    pub fn barycentric_simplified(a: Vec2, b: Vec2, c: Vec2, u: f32, v: f32, w: f32) -> Vec2 {
        a * u + b * v + c * w
    }

    /// Determines if a point lies within a triangle defined by three vertices.
    ///
    /// Uses barycentric coordinates to evaluate the points location relative to the triangle.
    ///
    /// # Parameters
    /// - `p`: The point to test.
    /// - `a`, `b`, `c`: The triangles corners.
    ///
    /// # Returns
    /// `true` if `p` lies inside or on the triangle, `false` otherwise.
    pub fn in_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
        let (u, v, w) = Vec2::barycentric(p, a, b, c);
        u >= 0.0 && v >= 0.0 && w >= 0.0
    }

    // ============= Comparison and Validity =============

    /// Returns `true` if `self` and `other` are approx equal within a default tolerance.
    ///
    /// This compares each component individually and returns `true` only if both are within epsilon.
    ///
    /// # Parameters
    /// - `other`: The other vector to compare.
    ///
    /// # Returns
    /// Boolean indicating approx equality.
    pub fn approx_eq(&self, other: Vec2) -> bool {
        epsilon_eq_default(self.x, other.x) && epsilon_eq_default(self.y, other.y)
    }

    /// Returns `true` if `self` and `other` are approx equal within a given epsilon tolerance.
    ///
    /// This compares each component individually and returns `true` only if both are within epsilon.
    ///
    /// # Parameters
    /// - `other`: The other vector to compare.
    /// - `epsilon`: The maximum allowed difference for each component.
    ///
    /// # Returns
    /// Boolean indicating approx equality.
    pub fn approx_eq_eps(&self, other: Vec2, epsilon: f32) -> bool {
        epsilon_eq(self.x, other.x, epsilon) && epsilon_eq(self.y, other.y, epsilon)
    }

    /// Checks whether both components of the vector are finite values.
    ///
    /// A component is considered finite if it is not `NaN` or infinite.
    ///
    /// # Returns
    /// `true` if both `x` and `y` are finite numbers.
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    /// Checks whether either component of the vector is `NaN`.
    ///
    /// # Returns
    /// `true` if `x` or `y` is `NaN`.
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan()
    }
}

// ============= Operator Overloads =============

/// Adds two vectors together component-wise.
impl Add for Vec2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

/// Adds a scalar to each component of the vector.
impl Add<f32> for Vec2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: f32) -> Self::Output {
        Self::new(self.x + rhs, self.y + rhs)
    }
}

/// Adds each component of a vector to a scalar.
impl Add<Vec2> for f32 {
    type Output = Vec2;
    #[inline]
    fn add(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self + rhs.x, self + rhs.y)
    }
}

/// Subtracts `rhs` from `self` component-wise.
impl Sub for Vec2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

/// Subtracts a scalar from each component of the vector.
impl Sub<f32> for Vec2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        Self::new(self.x - rhs, self.y - rhs)
    }
}

/// Subtracts each component of a vector from a scalar.
impl Sub<Vec2> for f32 {
    type Output = Vec2;
    #[inline]
    fn sub(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self - rhs.x, self - rhs.y)
    }
}

/// Multiplies two vectors together component-wise.
impl Mul for Vec2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.x * rhs.x, self.y * rhs.y)
    }
}

/// Multiplies each component of a vector by a scalar.
impl Mul<f32> for Vec2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

/// Multiplies a scalar by each component of a vector.
impl Mul<Vec2> for f32 {
    type Output = Vec2;
    #[inline]
    fn mul(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self * rhs.x, self * rhs.y)
    }
}

/// Divides `self` by `rhs` component-wise.
impl Div for Vec2 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self::new(self.x / rhs.x, self.y / rhs.y)
    }
}

/// Divides each component of a vector by a scalar.
impl Div<f32> for Vec2 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

/// Divides a scalar by each component of a vector.
impl Div<Vec2> for f32 {
    type Output = Vec2;
    #[inline]
    fn div(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self / rhs.x, self / rhs.y)
    }
}

/// Negates each component of the vector.
impl Neg for Vec2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y)
    }
}

// ============= Assignment Operator Overloads =============

impl AddAssign for Vec2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl AddAssign<f32> for Vec2 {
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        self.x += rhs;
        self.y += rhs;
    }
}

impl SubAssign for Vec2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl SubAssign<f32> for Vec2 {
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        self.x -= rhs;
        self.y -= rhs;
    }
}

impl MulAssign for Vec2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.x *= rhs.x;
        self.y *= rhs.y;
    }
}

impl MulAssign<f32> for Vec2 {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl DivAssign for Vec2 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.x /= rhs.x;
        self.y /= rhs.y;
    }
}

impl DivAssign<f32> for Vec2 {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

// ============= Trait Implementations =============

impl Default for Vec2 {
    /// Returns a `Vec2` with all components set to zero.
    #[inline]
    fn default() -> Self {
        Self::ZERO // Assumes Vec2::ZERO constant exists
    }
}

/// Checks whether two vectors are exactly equal.
/// Note: This performs an exact floating-point comparison. For tolerance-based
/// comparison, use the `approx` crate traits.
impl PartialEq for Vec2 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

/// Enables `v[index]` access. Panics if `index` is out of bounds.
impl Index<usize> for Vec2 {
    type Output = f32;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Vec2 index out of bounds: {}", index),
        }
    }
}

/// Enables mutable `v[index]` access. Panics if `index` is out of bounds.
impl IndexMut<usize> for Vec2 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Vec2 index out of bounds: {}", index),
        }
    }
}

/// Implements the `Display` trait for pretty-printing.
impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vec2({:.3}, {:.3})", self.x, self.y)
    }
}

// ============= Conversion Traits =============

/// Creates a `Vec2` from a tuple `(f32, f32)`.
impl From<(f32, f32)> for Vec2 {
    #[inline]
    fn from(t: (f32, f32)) -> Self {
        Self::new(t.0, t.1)
    }
}

/// Creates a tuple `(f32, f32)` from a `Vec2`.
impl From<Vec2> for (f32, f32) {
    #[inline]
    fn from(v: Vec2) -> Self {
        (v.x, v.y)
    }
}

/// Creates a `Vec2` from an array `[f32; 2]`.
impl From<[f32; 2]> for Vec2 {
    #[inline]
    fn from(arr: [f32; 2]) -> Self {
        Self::new(arr[0], arr[1])
    }
}

/// Creates an array `[f32; 2]` from a `Vec2`.
impl From<Vec2> for [f32; 2] {
    #[inline]
    fn from(v: Vec2) -> Self {
        [v.x, v.y]
    }
}

// ============= Approx Crate Implementations =============

/// Implements absolute difference equality comparison for `Vec2`.
impl approx::AbsDiffEq for Vec2 {
    type Epsilon = f32;

    #[inline]
    fn default_epsilon() -> f32 {
        f32::EPSILON
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: f32) -> bool {
        f32::abs_diff_eq(&self.x, &other.x, epsilon) && f32::abs_diff_eq(&self.y, &other.y, epsilon)
    }
}

/// Implements relative equality comparison for `Vec2`.
impl approx::RelativeEq for Vec2 {
    #[inline]
    fn default_max_relative() -> f32 {
        f32::EPSILON
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: f32, max_relative: f32) -> bool {
        f32::relative_eq(&self.x, &other.x, epsilon, max_relative)
            && f32::relative_eq(&self.y, &other.y, epsilon, max_relative)
    }
}
