use rand::Rng;

use crate::mat2::Mat2;
use crate::utils;
use crate::utils::epsilon_eq;
use crate::utils::epsilon_eq_default;
use crate::utils::is_near_zero_default;
use core::f32;
use std::fmt;

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

    /// Returns a 2D vector with both components set to zero.
    ///
    /// This is equivalent to the vector `(0.0, 0.0)`.
    ///
    /// # Returns
    /// A `Vec2` representing the zero vector.
    pub const fn zero() -> Vec2 {
        Vec2::new(0.0, 0.0)
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

    pub fn nan() -> Vec2 {
        Vec2::new(f32::NAN, f32::NAN)
    }

    pub fn infinity() -> Vec2 {
        Vec2::new(f32::INFINITY, f32::INFINITY)
    }

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
    /// a `Vec2` representing the sign of each component.
    pub fn signum(self) -> Vec2 {
        Vec2::new(self.x.signum(), self.y.signum())
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
    /// use game_math::vec2::Vec2;
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
    /// use game_math::vec2::Vec2;
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
            Vec2::zero()
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
    /// use game_math::vec2::Vec2;
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
    /// use game_math::vec2::Vec2;
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
    /// use game_math::vec2::Vec2;
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
    /// use game_math::vec2::Vec2;
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
    /// use game_math::vec2::Vec2;
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
    /// use game_math::vec2::Vec2;
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
        *self = Mat2::from_rotation(angle_rad) * *self;
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
    /// use game_math::vec2::Vec2;
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
use std::ops::Add;

/// Adds two vectors together component-wise
///
/// # Parameters
/// - `rhs`: Other vector.
///
/// # Returns
/// `Vec2` which is the sum of `self` and `other`.
impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self.x + rhs.x, self.y + rhs.y)
    }
}

/// Adds each element of the vector with the scalar.
///
/// # Parameters
/// - `scalar`: The float to use.
///
/// # Returns
/// `Vec2` which is the sum `(self.x + scalar, self.y + scalar)`.
impl Add<f32> for Vec2 {
    type Output = Vec2;
    fn add(self, scalar: f32) -> Self::Output {
        Vec2::new(self.x + scalar, self.y + scalar)
    }
}

/// Adds the float to each element of a given vector.
///
/// # Parameters
/// - `v`: The vector to add.
///
/// # Returns
/// `Vec2` which is the sum `(self + v.x, self + v.y)`.
impl Add<Vec2> for f32 {
    type Output = Vec2;
    fn add(self, v: Vec2) -> Self::Output {
        Vec2::new(self + v.x, self + v.y)
    }
}
use std::ops::Sub;

/// Subtracts `rhs` from `self` component-wise.
///
/// # Parameters
/// - `rhs`: The vector on the right hand side of the calculation.
///
/// # Returns
/// `Vec2` which is the sum `(self.x - rhs.x, self.y - rhs.y)`.
impl Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self.x - rhs.x, self.y - rhs.y)
    }
}

/// Subtracts `float` from `self` component-wise.
///
/// # Parameters
/// - `scalar`: The float to use.
///
/// # Returns
/// `Vec2` which is the sum `(self.x - scalar, self.y - scalar)`.
impl Sub<f32> for Vec2 {
    type Output = Vec2;
    fn sub(self, scalar: f32) -> Self::Output {
        Vec2::new(self.x - scalar, self.y - scalar)
    }
}

/// Subtracts `self` from each component of `v`.
///
/// # Parameters
/// - `v`: The vector to use.
///
/// # Returns
/// `Vec2` which is the sum `(scalar - v.x, scalar - v.y)`.
impl Sub<Vec2> for f32 {
    type Output = Vec2;
    fn sub(self, v: Vec2) -> Self::Output {
        Vec2::new(self - v.x, self - v.y)
    }
}

use std::ops::Mul;

/// Multiplies two vectors together component-wise.
///
/// # Parameters
/// - `rhs`: The vector on the right hand side of the calculation.
///
/// # Returns
/// `Vec2` which is the product of `self * rhs`.
impl Mul for Vec2 {
    type Output = Vec2;
    fn mul(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self.x * rhs.x, self.y * rhs.y)
    }
}

/// Multiplies each component of a vector by a scalar value.
///
/// # Parameters
/// - `scalar`: The float to use.
///
/// # Returns
/// `Vec2` which is the product of `(self.x * scalar, self.y * scalar)`.
impl Mul<f32> for Vec2 {
    type Output = Vec2;
    fn mul(self, scalar: f32) -> Self::Output {
        Vec2::new(self.x * scalar, self.y * scalar)
    }
}

/// Multiplies a float by each component of a vector.
///
/// # Parameters
/// - `vec`: The vector to use.
///
/// # Returns
/// `Vec2` which is the product of `(self * vec.x, self * vec.y)`.
impl Mul<Vec2> for f32 {
    type Output = Vec2;
    fn mul(self, vec: Vec2) -> Vec2 {
        Vec2::new(vec.x * self, vec.y * self)
    }
}

use std::ops::Div;

/// Divides vector `self` from `rhs` component-wise.
///
/// # Parameters
/// - `rhs`: The right hand side of the calculation.
///
/// # Returns
/// `Vec2` which is the result of `(self.x / rhs.x, self.y / rhs.y)`.
impl Div for Vec2 {
    type Output = Vec2;
    fn div(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self.x / rhs.x, self.y / rhs.y)
    }
}

/// Divides each component of a vector by `scalar`.
///
/// # Parameters
/// - `scalar`: The float to use.
///
/// # Returns
/// `Vec2` which is the result of `(self.x / scalar, self.y / scalar)`.
impl Div<f32> for Vec2 {
    type Output = Vec2;
    fn div(self, scalar: f32) -> Self::Output {
        Vec2::new(self.x / scalar, self.y / scalar)
    }
}

/// Divides scalar by each component of a vector.
///
/// # Parameters
/// - `v`: Vector to use.
///
/// # Returns
/// `Vec2` which is the result of `(self / v.x, self / v.y)`.
impl Div<Vec2> for f32 {
    type Output = Vec2;
    fn div(self, v: Vec2) -> Self::Output {
        Vec2::new(self / v.x, self / v.y)
    }
}

use std::ops::Neg;

/// Negates all components in a vector to be their negative values.
///
/// This could turn a 1.0 into a -1.0 or vice versa.
///
/// # Returns
/// `Vec2` where all components are negated.
impl Neg for Vec2 {
    type Output = Self;
    fn neg(self) -> Self {
        Vec2::new(-self.x, -self.y)
    }
}

use std::cmp::PartialEq;

/// Checks whether two vectors are exactly equal.
///
/// This doesn't have an epsilon meaning floating point errors may happen.
///
/// # Parameters
/// - `other`: The other vector to compare to.
///
/// # Returns
/// `true` if `self.x == other.x` && `self.y == other.y`. Otherwise `false`.
impl PartialEq for Vec2 {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

/// Implements absolute difference equality comparison for `Vec2`.
///
/// This trait allows comparing two vectors for approx equality
/// using a fixed absolute epsilon value. Useful when working with
/// floating point precision issues, such as physics or graphics code.
///
/// # Example
/// ```rust
/// use approx::AbsDiffEq;
/// use game_math::vec2::Vec2;
/// let a = Vec2::new(1.0, 2.0);
/// let b = Vec2::new(1.0 + 1e-7, 2.0 - 1e-7);
/// assert!(a.abs_diff_eq(&b, 1e-6));
/// ```
impl approx::AbsDiffEq for Vec2 {
    type Epsilon = f32;

    fn default_epsilon() -> f32 {
        f32::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: f32) -> bool {
        f32::abs_diff_eq(&self.x, &other.x, epsilon) && f32::abs_diff_eq(&self.y, &other.y, epsilon)
    }
}

/// implements relative equality comparison for `Vec2`.
///
/// This trait compares values based on their relative difference
/// instead of absolute. Its useful when values can scale significantly
/// and a fixed epsilon isn't sufficient.
///
/// # Example
/// ```rust
/// use approx::RelativeEq;
/// use game_math::vec2::Vec2;
/// let a = Vec2::new(100.0, 200.0);
/// let b = Vec2::new(100.00001, 200.00001);
/// assert!(a.relative_eq(&b, 1e-6, 1e-6));
/// ```
impl approx::RelativeEq for Vec2 {
    fn default_max_relative() -> f32 {
        f32::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: f32, max_relative: f32) -> bool {
        f32::relative_eq(&self.x, &other.x, epsilon, max_relative)
            && f32::relative_eq(&self.y, &other.y, epsilon, max_relative)
    }
}

use std::ops::{Index, IndexMut};

/// Enables v[index] access.
///
/// # Panics
/// If index >= 2.
impl Index<usize> for Vec2 {
    type Output = f32;
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Vec2 index out of bounds: {}", index),
        }
    }
}

/// Enables mutable v[index] access.
///
/// # Panics
/// If index >= 2.
impl IndexMut<usize> for Vec2 {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Vec2 index out of bounds: {}", index),
        }
    }
}

/// Method to turn tuples into a vector.
///
/// # Parameters
/// - `t`: The tuple to convert to a vector.
///
/// # Returns
/// `Vec2` which is `(t.0, t.1)`.
impl From<(f32, f32)> for Vec2 {
    fn from(t: (f32, f32)) -> Self {
        Vec2::new(t.0, t.1)
    }
}

/// Method to turn vectors into tuples.
///
/// # Parameters
/// - `v`: The vector to be converted to a tuple.
///
/// # Returns
/// `tuple` which is `(v.x, v.y)`.
impl From<Vec2> for (f32, f32) {
    fn from(v: Vec2) -> Self {
        (v.x, v.y)
    }
}

/// Method to turn an array into a vector.
///
/// # Parameters
/// - `arr`: Array to be converted.
///
/// # Returns
/// `Vec2` which is `(arr[0], arr[1])`.
impl From<[f32; 2]> for Vec2 {
    fn from(arr: [f32; 2]) -> Self {
        Vec2::new(arr[0], arr[1])
    }
}

/// Method to turn a vector into an array.
///
/// # Parameters
/// - `v`: The vector to be converted.
///
/// # Returns
/// `array` which is `[v.x, v.y]`.
impl From<Vec2> for [f32; 2] {
    fn from(v: Vec2) -> [f32; 2] {
        [v.x, v.y]
    }
}

/// Method to correctly display a vector.
///
/// # Parameters
/// - `f`: Formatter to use.
///
/// # Returns
/// `Result` which is a pretty-printed version of the vector.
impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vec2({}, {})", self.x, self.y)
    }
}
