use rand::Rng;

use crate::mat4::Mat4;
use crate::utils;
use crate::utils::epsilon_eq;
use crate::utils::epsilon_eq_default;
use crate::utils::is_near_zero_default;
use crate::vec::Vec3;

/// A 4D vector with x, y, z, and w components.
#[derive(Debug, Clone, Copy)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    // ============= Construction and Conversion =============

    /// Creates a 4D vector with the given `x`, `y`, `z`, and `w` components.
    ///
    /// # Parameters
    /// - `x`: The x component of the vector.
    /// - `y`: The y component of the vector.
    /// - `z`: The z component of the vector.
    /// - `w`: The w component of the vector.
    ///
    /// # Returns
    /// A new `Vec4` instance with the specified components.
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
        Vec4 { x, y, z, w }
    }

    /// Returns a 4D vector where all components are set to zero.
    ///
    /// This is equivalent to the vector `(0.0, 0.0, 0.0, 0.0)`.
    ///
    /// # Returns
    /// A `Vec4` representing the zero vector.
    pub fn zero() -> Vec4 {
        Vec4::new(0.0, 0.0, 0.0, 0.0)
    }

    /// Converts the vector into an array of 4 `f32` components `[x, y, z, w]`.
    ///
    /// # Returns
    /// An array [f32; 4].
    pub fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// Converts an array into a vector.
    ///
    /// # Parameters
    /// - `arr`: The array to convert.
    ///
    /// # Returns
    /// `Vec4` which is equivalent to `(arr[0], arr[1], arr[2], arr[3])`.
    pub fn from_array(arr: [f32; 4]) -> Vec4 {
        Vec4::new(arr[0], arr[1], arr[2], arr[3])
    }

    /// Converts the vector into a tuple `(f32, f32, f32, f32)`.
    ///
    /// # Returns
    /// `(f32, f32, f32, f32)` which is equivalent to (self.x, self.y, self.z, self.w).
    pub fn to_tuple(self) -> (f32, f32, f32, f32) {
        (self.x, self.y, self.z, self.w)
    }

    /// Converts a tuple to a vector.
    ///
    /// # Parameters
    /// - `t`: Tuple `(f32, f32, f32, f32)` to be converted.
    ///
    /// # Returns
    /// `Vec4` which is equivalent to `(t.0, t.1, t.2, t.3)`.
    pub fn from_tuple(tuple: (f32, f32, f32, f32)) -> Vec4 {
        Vec4::new(tuple.0, tuple.1, tuple.2, tuple.3)
    }

    /// Creates a new vector where all components are NaN.
    ///
    /// # Returns
    /// `Vec4`.
    pub fn nan() -> Vec4 {
        Vec4::new(f32::NAN, f32::NAN, f32::NAN, f32::NAN)
    }

    /// Creates a new vector where all components are INFINITY.
    ///
    /// # Returns
    /// `Vec4`.
    pub fn infinity() -> Vec4 {
        Vec4::new(f32::INFINITY, f32::INFINITY, f32::INFINITY, f32::INFINITY)
    }

    /// Method to create a Vec4 from a Vec3.
    ///
    /// # Parameters
    /// - `v`: Vec2 to use for `x`, `y`, and `z` components.
    /// - `w`: f32 to use for the `w` component.
    ///
    /// # Returns
    /// `Vec4`.
    pub fn from_vec3_w(v: Vec3, w: f32) -> Vec4 {
        Vec4::new(v.x, v.y, v.z, w)
    }

    /// Returns a Vec3 that is the `x`, `y`, and `z` components of `self`.
    ///
    /// # Returns
    /// `Vec3`.
    pub fn xyz(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    // ============= Math Utilities =============

    /// Returns a new vector with each component replaced by its absolute value.
    ///
    /// This is useful for obtaining the magnitude of each axis regardless of sign.
    ///
    /// # Returns
    /// A `Vec4` where `x`, `y`, `z`, and `w` are the absolute vaklues of the original vectors components.
    pub fn abs(self) -> Vec4 {
        Vec4::new(self.x.abs(), self.y.abs(), self.z.abs(), self.w.abs())
    }

    /// Returns a new vector where each component is replaced by its sign.
    ///
    /// The sign of each component can be `-1.0`, `0.0`, or `1.0` depending on whether the
    /// component is negative, zero, or positive respectively.
    ///
    /// # Returns
    /// A `Vec4` representing the sign of each component.
    pub fn signum(self) -> Vec4 {
        Vec4::new(
            self.x.signum(),
            self.y.signum(),
            self.z.signum(),
            self.w.signum(),
        )
    }

    /// Clamps each component of the vector between the specified `min` and `max` values.
    ///
    /// # Parameters
    /// - `min`: The minimum allowed value for each component.
    /// - `max`: The maximum allowed value for each component.
    ///
    /// # Returns
    /// A new `Vec4` where each component is limited to the range `[min, max]`.
    pub fn clamp(self, min: f32, max: f32) -> Vec4 {
        Vec4::new(
            utils::clamp(self.x, min, max),
            utils::clamp(self.y, min, max),
            utils::clamp(self.z, min, max),
            utils::clamp(self.w, min, max),
        )
    }

    /// Returns the component-wise minimum between `self` and another vector.
    ///
    /// # Parammeters
    /// - `other`: The other vector.
    ///
    /// # Returns
    /// A `Vec4` with each component being the minimum of the corresponding components.
    ///
    /// ```rust
    /// use omelet::vec4::Vec4;
    /// let a = Vec4::new(1.0, 4.0, 2.0, 1.0);
    /// let b = Vec4::new(3.0, 1.0, 1.0, 5.0);
    /// assert_eq!(a.min(b), Vec4::new(1.0, 1.0, 1.0, 1.0));
    /// ```
    pub fn min(self, other: Vec4) -> Vec4 {
        Vec4::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
            self.w.min(other.w),
        )
    }

    /// Returns the component-wise maximum between `self` and another vector.
    ///
    /// # Parammeters
    /// - `other`: The other vector.
    ///
    /// # Returns
    /// A `Vec4` with each component being the maximum of the corresponding components.
    ///
    /// ```rust
    /// use omelet::vec4::Vec4;
    /// let a = Vec4::new(1.0, 4.0, 2.0, 1.0);
    /// let b = Vec4::new(3.0, 1.0, 1.0, 5.0);
    /// assert_eq!(a.max(b), Vec4::new(3.0, 4.0, 2.0, 5.0));
    /// ```
    pub fn max(self, other: Vec4) -> Vec4 {
        Vec4::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
            self.w.max(other.w),
        )
    }

    /// Computes the **scalar 4D triple product** for four vectors:
    /// `a · (b × c × d)` - interpreted as the **determinant of the 4x4 matrix**
    /// whose columns are the vectors `a`, `b`, `c`, and `d`.
    ///
    /// This is used to compute **signed 3D hypervolume** (also called the 4-parallelotope volume).
    ///
    /// # Parameters
    /// - `a`, `b`, `c`, `d`: 4D vectors used to form a 4x4 matrix.
    /// 
    /// # Returns
    /// Signed scalar volume (can be negative depending on handedness).
    pub fn triple_product_4d(a: Vec4, b: Vec4, c: Vec4, d: Vec4) -> f32 {
        let m = Mat4::new(
            Vec4::new(a.x, b.x, c.x, d.x),
            Vec4::new(a.y, b.y, c.y, d.y),
            Vec4::new(a.z, b.z, c.z, d.z),
            Vec4::new(a.w, b.w, c.w, d.w),
        );

        // Compute determinant using Laplace expansion or precomputed formula
        // This expansion is hardcoded for performance
        m[0][0]
            * (m[1][1] * (m[2][2] * m[3][3] - m[2][3] * m[3][2])
                - m[1][2] * (m[2][1] * m[3][3] - m[2][3] * m[3][1])
                + m[1][3] * (m[2][1] * m[3][2] - m[2][2] * m[3][1]))
            - m[0][1]
                * (m[1][0] * (m[2][2] * m[3][3] - m[2][3] * m[3][2])
                    - m[1][2] * (m[2][0] * m[3][3] - m[2][3] * m[3][0])
                    + m[1][3] * (m[2][0] * m[3][2] - m[2][2] * m[3][0]))
            + m[0][2]
                * (m[1][0] * (m[2][1] * m[3][3] - m[2][3] * m[3][1])
                    - m[1][1] * (m[2][0] * m[3][3] - m[2][3] * m[3][0])
                    + m[1][3] * (m[2][0] * m[3][1] - m[2][1] * m[3][0]))
            - m[0][3]
                * (m[1][0] * (m[2][1] * m[3][2] - m[2][2] * m[3][1])
                    - m[1][1] * (m[2][0] * m[3][2] - m[2][2] * m[3][0])
                    + m[1][2] * (m[2][0] * m[3][1] - m[2][1] * m[3][0]))
    }

    /// Returns the **unsigned 4D hypervolume** formed by the vectors `a`, `b`, `c`, and `d`,
    /// using the **absolute value** of the 4D scalar triple product.
    /// 
    /// # parameters
    /// - `a`, `b`, `c`, `d`: 4D vectors forming the 4D volume.
    /// 
    /// # Returns
    /// The absolute hypervolume (alsways non-negative).
    pub fn hypervolume_4d(a: Vec4, b: Vec4, c: Vec4, d: Vec4) -> f32 {
        Vec4::triple_product_4d(a, b, c, d).abs()
    }

    /// Returns a **unit-length vector perpendicular** to the current vector in 4D space.
    /// 
    /// Chooses a perpendicular direction by nulling the smallest component and rotating others.
    /// If the vector is degenerate (zero vector), a defauly fallback of (1, 0, 0, 0) is returned.
    /// 
    /// # Returns
    /// A normalized perpendicular Vec4.
    pub fn perpendicular(self) -> Vec4 {
        // Pick a perpendicular vector by zeroing the smallest component
        let mut v = if self.x.abs() <= self.y.abs()
            && self.x.abs() <= self.z.abs()
            && self.x.abs() <= self.w.abs()
        {
            Vec4::new(0.0, -self.z, self.y, -self.w)
        } else if self.y.abs() <= self.z.abs() && self.y.abs() <= self.w.abs() {
            Vec4::new(-self.z, 0.0, self.x, -self.w)
        } else if self.z.abs() <= self.w.abs() {
            Vec4::new(-self.y, self.x, 0.0, -self.w)
        } else {
            Vec4::new(-self.y, self.x, -self.z, 0.0)
        };
        v = v.normalize_or_zero();
        if v.is_zero() {
            Vec4::new(1.0, 0.0, 0.0, 0.0) // Fallback
        } else {
            v
        }
    }

    // ============= Magnitude and Normalization =============

    /// Calculates and returns the magnitude (length) of the vector.
    ///
    /// The length is computed as the squre root of the sum of sqaures of `x`, `y`, `z`, and `w`.
    ///
    /// # Returns
    /// A `f32` representing the Euclidean length of the vector.
    ///
    /// Note: This calculation uses sqrt(), which can be taxing on the system.
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    /// Computes and returns the squared magnitude (length) of the vector.
    ///
    /// This avoids the expensive square root calculation used in `length()`,
    /// and it useful when only relative length or comparisons are needed.
    ///
    /// # Returns
    /// A `f32` representing the squared length `(x² + y² + z² + w²)`.
    pub fn squared_length(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    /// Returns a normalized (unit length) vector in the same direction as `self`.
    ///
    /// If the length is near 0 within the default tolerance, then it will return a zero vector.
    ///
    /// # Returns
    /// `Vec4` either normalized or zero.
    pub fn normalize_or_zero(&self) -> Vec4 {
        let len = self.length();
        if len == 0.0 {
            Vec4::new(0.0, 0.0, 0.0, 0.0)
        } else {
            Vec4::new(self.x / len, self.y / len, self.z / len, self.w / len)
        }
    }

    /// Attempts to return a normalized version of the vector, or `None` if the length
    /// is zero or nearly zero.
    ///
    /// # Returns
    /// - `Some(Vec4)` containing the normalized vector if length > epsilon.
    /// - `None` if vector length is zero or near zero (within `1e-6`).
    pub fn try_normalize(&self) -> Option<Vec4> {
        let len = self.length();
        if is_near_zero_default(len) {
            None
        } else {
            Some(Vec4::new(
                self.x / len,
                self.y / len,
                self.z / len,
                self.w / len,
            ))
        }
    }

    /// Returns a normalized (unit length) vector in the same direction as `self`.
    ///
    /// # Panics
    /// This function panics if the vector length is zero to prevent division by zero.
    ///
    /// # Returns
    /// A `Vec4` with length 1 pointing in the same direction.
    pub fn normalize(&self) -> Vec4 {
        let len = self.length();
        assert!(
            !is_near_zero_default(len),
            "Cannot normalize zero length vector"
        );
        Vec4::new(self.x / len, self.y / len, self.z / len, self.w / len)
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
    /// The dot product is defined as `x1 * x2 + y1 * y2 + z1 * z2 + w1 * w2` and measures the similarity
    /// of the two vectors directions.
    ///
    /// # Parameters
    /// - `other`: The other `Vec4` to perfom the dot with.
    ///
    /// # Returns
    /// A scalar `f32` value representing the dot product.
    pub fn dot(&self, other: Vec4) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    /// Computes the cross product of two vectors.
    ///
    /// The cross product is defined as:
    /// ```text
    /// x = v1.y * v2.z - v1.z * v2.y
    /// y = v1.z * v2.x - v1.x * v2.z
    /// z = v1.x * v2.y - v1.y * v2.x
    /// ```
    ///
    /// # Parameters
    /// - `other`: The other vector in the equation.
    ///
    /// # Returns
    /// A new `Vec4` which is the result of the cross product between `self` and `other`.
    ///
    /// Note: This calculation ignores `w`.
    pub fn cross_xyz(&self, other: Vec4) -> Vec4 {
        Vec4::new(
            self.y * other.z - self.z * other.y, //x
            self.z * other.x - self.x * other.z, //y
            self.x * other.y - self.y * other.x, //z
            self.w,
        )
    }

    /// Computes the angle (in radians) between this vector and another.
    /// 
    /// # Parameters
    /// - `other`: A 4D vector to compare against.
    /// 
    /// # Returns
    /// Angle in radians between the vectors. Returns 0.0 if either vector is zero-length.
    pub fn angle_to(self, other: Vec4) -> f32 {
        let dot = self.dot(other);
        let len_product = self.length() * other.length();
        if epsilon_eq_default(len_product, 0.0) {
            return 0.0;
        }
        (dot / len_product).clamp(-1.0, 1.0).acos()
    }

    /// Calculates the angle in radians between `self` and another vector.
    ///
    /// The angle is in the range `[0, π]`.
    ///
    /// # Parameters
    /// - `other`: The other vector.
    ///
    /// # Returns
    /// The angle `f32` between the two vectors in radians.
    pub fn angle_between_radians(a: Vec4, b: Vec4) -> f32 {
        a.dot(b).clamp(-1.0, 1.0).acos()
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
    pub fn angle_between_degrees(a: Vec4, b: Vec4) -> f32 {
        utils::radians_to_degrees(a.dot(b).clamp(-1.0, 1.0).acos())
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
    /// A `Vec4` representing the point `t` fraction between `self` and `b`.
    pub fn lerp(self, target: Vec4, t: f32) -> Vec4 {
        self * (1.0 - t) + target * t
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
    /// A clamped interpolated `Vec4`.
    pub fn lerp_clamped(self, target: Vec4, t: f32) -> Vec4 {
        let t = utils::clamp(t, 0.0, 1.0);
        self.lerp(target, t)
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
    pub fn lerp_between(a: Vec4, b: Vec4, t: f32) -> Vec4 {
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
    pub fn lerp_between_clamped(a: Vec4, b: Vec4, t: f32) -> Vec4 {
        let t = utils::clamp(t, 0.0, 1.0);
        Vec4::lerp_between(a, b, t)
    }

    /// Performs spherical linear interpolation (SLERP) between two 4D vectors.
    ///
    /// SLERP smoothly interpolates between two directions over the surface of a sphere,
    /// preserving constant angular velocity and the length of the interpolated vector.
    /// This version handles vectors of different magnitudes and falls back to
    /// linear interpolation if the angle between them is too small.
    ///
    /// # Parameters
    /// - `a`: Starting vector.
    /// - `b`: Ending vector.
    /// - `t`: Interpolation factor in `[0.0, 1.0]`.
    ///
    /// # Returns
    /// A vector that represents the spherical interpolation between `a` and `b` at `t`.
    ///
    /// # Behavior
    /// - If either `a` or `b` is near-zero length, it blends based on magnitude.
    /// - If vectors are nearly parallel (dot > 1 - epsilon), falls back to clamped lerp.
    /// - Interpolates both direction and magnitude.
    ///
    /// # Example
    /// ```rust
    /// use omelet::vec4::Vec4;
    /// let a = Vec4::new(1.0, 0.0, 0.0, 0.0);
    /// let b = Vec4::new(0.0, 1.0, 0.0, 0.0);
    /// let halfway = Vec4::slerp(a, b, 0.5);
    /// ```
    pub fn slerp(a: Vec4, b: Vec4, t: f32) -> Vec4 {
        // Handle zero vectors
        if a.length() < 1e-6 {
            return b * t;
        }
        if b.length() < 1e-6 {
            return a * (1.0 - t);
        }

        let a_norm = a.normalize_or_zero();
        let b_norm = b.normalize_or_zero();
        let dot = a_norm.dot(b_norm).clamp(-1.0, 1.0);

        if dot > 1.0 - 1e-6 {
            return Vec4::lerp(a, b, t);
        }

        let theta = dot.acos();
        let sin_theta = theta.sin();
        let wa = ((1.0 - t) * theta).sin() / sin_theta;
        let wb = (t * theta).sin() / sin_theta;

        (a_norm * wa + b_norm * wb) * (a.length() * (1.0 - t) + b.length() * t)
    }

    /// Performs spherical linear interpolation (SLERP) between two 4D vectors
    /// using angle interpolation with shortest path logic.
    ///
    /// This function is conceptually similar to [`slerp`], but is optimised for
    /// interpolating orientations with the shortest arc. It assumes the vectors represent directions,
    /// and normalizes them before interpolation.
    ///
    /// # Parameters
    /// - `a`: Starting vector.
    /// - `b`: Ending vector.
    /// - `t`: Interpolation factor in `[0.0, 1.0]`.
    ///
    /// # Returns
    /// A vectopr representing the interpolated direction, scaled to smoothly blend
    /// the lengths of `a` and `b`.
    ///
    /// # Behavior
    /// - Normalizes both vectors before interpolation.
    /// - If either vector is near-zero, falls back to linear interpolation.
    /// - If the angle between vectors is too small, falls back to clamped lerp.
    ///
    /// # Example
    /// ```rust
    /// use omelet::vec4::Vec4;
    /// let start = Vec4::new(1.0, 0.0, 0.0, 0.0);
    /// let end = Vec4::new(0.0, 1.0, 0.0, 0.0);
    /// let mid = Vec4::slerp_angle(start, end, 0.5);
    /// ```
    pub fn slerp_angle(a: Vec4, b: Vec4, t: f32) -> Vec4 {
        // Handle zero vectors
        if a.length() < 1e-6 || b.length() < 1e-6 {
            return Vec4::lerp_between_clamped(a, b, t);
        }

        let a_norm = a.normalize();
        let b_norm = b.normalize();
        let dot = a_norm.dot(b_norm).clamp(-1.0, 1.0);

        // If the vectors are nearly parallel, use linear interpolation
        if dot > 1.0 - 1e-6 {
            return Vec4::lerp_between_clamped(a, b, t);
        }

        // Calculate the angle between the vectors
        let theta = dot.acos();

        // Interpolate using the slerp formula
        let sin_theta = theta.sin();
        let wa = ((1.0 - t) * theta).sin() / sin_theta;
        let wb = (t * theta).sin() / sin_theta;

        // Interpolate both direction and magnitude
        (a_norm * wa + b_norm * wb) * (a.length() * (1.0 - t) + b.length() * t)
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
    pub fn project(&self, onto: Vec4) -> Vec4 {
        let denominator = onto.dot(onto);
        if denominator == 0.0 {
            Vec4::new(0.0, 0.0, 0.0, 0.0)
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
    /// A `Vec4` representing the component of `self` perpendicular to `other`.
    pub fn reject(&self, other: Vec4) -> Vec4 {
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
    /// A `Vec4` reflected.
    pub fn reflect(&self, normal: Vec4) -> Vec4 {
        *self - normal * (2.0 * self.dot(normal))
    }

    /// Returns the mirrored vector over a given normal.
    ///
    /// # Parameters
    /// - `normal`: The normal to mirror the vector over.
    ///
    /// # Returns
    /// `Vec4` which is mirrored over `normal`.
    pub fn mirror(&self, normal: Vec4) -> Vec4 {
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
    pub fn distance(&self, other: Vec4) -> f32 {
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
    pub fn squared_distance(&self, other: Vec4) -> f32 {
        (*self - other).squared_length()
    }

    /// Returns the normalized direction vector pointing from `self` towards `other`.
    ///
    /// # Parameters
    /// - `other`: The target vector.
    ///
    /// # Returns
    /// A unit length `Vec4` pointing from `self` to `other`.
    ///
    /// # Panics
    /// Panics if the direction vector length is zero (i.e. both vectors equal).
    pub fn direction_to(&self, other: Vec4) -> Vec4 {
        let delta = other - *self;
        delta.normalize_or_zero()
    }

    /// Returns the raw direction vector from `self` to `other` without normalization.
    ///
    /// # Parameters
    /// - `other`: The target vector.
    ///
    /// # Returns
    /// The difference vector `other - self`.
    pub fn direction_to_raw(&self, other: Vec4) -> Vec4 {
        other - *self
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
    /// A new `Vec4` moved towards the target by `max_delta` or exactly `target` if close enough.
    ///
    /// # Example
    /// ```rust
    /// use omelet::vec4::Vec4;
    /// let current = Vec4::new(0.0, 0.0, 0.0, 0.0);
    /// let target = Vec4::new(10.0, 0.0, 0.0, 0.0);
    /// let moved = Vec4::move_towards(current, target, 3.0);
    /// assert_eq!(moved, Vec4::new(3.0, 0.0, 0.0, 0.0));
    /// ```
    pub fn move_towards(current: Vec4, target: Vec4, max_delta: f32) -> Vec4 {
        let delta = target - current;
        let distance = delta.length();

        if distance <= max_delta || distance < f32::EPSILON {
            target
        } else {
            current + delta / distance * max_delta
        }
    }

    // ============= Geometry =============

    /// Returns two orthonormal vectors.
    ///
    /// Given trwo non-parallel vectors, returns two orthonormal vectors:
    /// the first is a normalized version of the input `a` and the second is
    /// perpendicular to `a` and lies in the same plane as `a` and `b`.
    ///
    /// # Parameters
    /// - `a`: First input vector.
    /// - `b`: Second input vector to orthonormalize relative to `a`.
    ///
    /// # Returns
    /// Tuple `(a_orthonormal, b_orthonormal)` where both vectors are unit length
    /// and perpendicular to each other.
    ///
    /// # Panics
    /// Panics if input vectors are zero or nearly parallel.
    pub fn orthonormalize(a: Vec4, b: Vec4) -> (Vec4, Vec4) {
        let a_norm = a.normalize();
        let projection = b.project(a_norm);
        let b_orthogonal = b - projection;
        let b_norm = b_orthogonal.normalize();
        (a_norm, b_norm)
    }

    /// Returns a vector rotated around an axis by a specific angle.
    ///
    /// # Parameters
    /// - `axis`: Axis to rotate around.
    /// - `angle`: Angle to rotate by.
    ///
    /// # Returns
    /// A `Vec4` rotated around the `axis` by `angle` using Rodrigues rotation formula.
    pub fn rotate_around_axis(&self, axis: Vec4, angle: f32) -> Vec4 {
        let axis = axis.normalize();
        let cos = angle.cos();
        let sin = angle.sin();
        *self * cos + axis.cross_xyz(*self) * sin + axis * axis.dot(*self) * (1.0 - cos)
    }

    /// Rotates this vector **within a 2D plane** formed by two axis indices in 4D space. 
    /// 
    /// Applies a rotation by the given angle in radians in the plane defined by axis `a` and `b`.
    /// 
    /// # Parameters
    /// - `a`, `b`: Axis indices (0 = x, 1 = y, 2 = z, 3 = w); must be distinct.
    /// - `angle`: Angle of rotation in radians.
    /// 
    /// # Panics
    /// If `a` or `b` are out of bounds or equal.
    /// 
    /// # Returns
    /// The rotated Vec4.
    pub fn rotate_in_plane(self, a: usize, b: usize, angle: f32) -> Vec4 {
        assert!(
            a < 4 && b < 4 && a != b,
            "Invalid acis indices for 4D plane rotation"
        );

        let sin = angle.sin();
        let cos = angle.cos();

        let mut components = [self.x, self.y, self.z, self.w];
        let (va, vb) = (components[a], components[b]);

        components[a] = va * cos - vb * sin;
        components[b] = va * sin + vb * cos;

        Vec4::new(components[0], components[1], components[2], components[3])
    }

    // ============= Random =============

    /// Returns a randomly generated unit vector using rand crate.
    ///
    /// # Returns
    /// `Vec4` which each component is randomly generated in range `[0.0, 1.0]`.
    pub fn random_unit_vector() -> Vec4 {
        let mut rng = rand::rng();
        loop {
            let x = rng.random_range(-1.0..=1.0);
            let y = rng.random_range(-1.0..=1.0);
            let z = rng.random_range(-1.0..=1.0);
            let w = rng.random_range(-1.0..=1.0);

            let v = Vec4::new(x, y, z, w);
            let len_sq = v.squared_length();
            if len_sq > 0.0 && len_sq <= 1.0 {
                return v / len_sq.sqrt();
            }
        }
    }

    /// Generates a random point inside the 4D unit hypersphere (radius ≤ 1).
    ///
    /// This uses rejection sampling within a [-1, 1]^4 hypercube.
    /// The returned vector lies inside or on the boundary of the unit 4D sphere.
    /// 
    /// # Returns
    /// A randomly generated Vec4 within the unit 4D sphere.
    pub fn random_in_unit_sphere() -> Vec4 {
        let mut rng = rand::rng();
        loop {
            let x = rng.random_range(-1.0..=1.0);
            let y = rng.random_range(-1.0..=1.0);
            let z = rng.random_range(-1.0..=1.0);
            let w = rng.random_range(-1.0..=1.0);
            let v = Vec4::new(x, y, z, w);
            if v.squared_length() <= 1.0 {
                return v;
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
    /// A tupel `(u, v, w)` of barycentric weight where `u + v + w = 1`.
    ///
    /// Note: Works properly if a, b, c, p all lie in the same plane.
    ///
    /// /// # Example
    /// ```rust
    /// use omelet::vec4::Vec4;
    /// let p = Vec4::new(2.0, 1.0, 0.0, 0.0);
    /// let a = Vec4::new(0.0, 0.0, 0.0, 0.0);
    /// let b = Vec4::new(4.0, 0.0, 0.0, 0.0);
    /// let c = Vec4::new(0.0, 4.0, 0.0, 0.0);
    /// let (u, v, w) = Vec4::barycentric(p, a, b, c);
    /// assert!((u + v + w - 1.0).abs() < 1e-6);
    /// ```
    pub fn barycentric(p: Vec4, a: Vec4, b: Vec4, c: Vec4) -> (f32, f32, f32) {
        let v0 = b - a;
        let v1 = c - a;
        let v2 = p - a;

        let d00 = v0.dot(v0);
        let d01 = v0.dot(v1);
        let d11 = v1.dot(v1);
        let d20 = v2.dot(v0);
        let d21 = v2.dot(v1);

        let denom = d00 * d11 - d01 * d01;
        if is_near_zero_default(denom) {
            return (0.0, 0.0, 0.0); // degenerate triangle
        }

        let v = (d11 * d20 - d01 * d21) / denom;
        let w = (d00 * d21 - d01 * d20) / denom;
        let u = 1.0 - v - w;

        (u, v, w)
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
    pub fn in_triangle(p: Vec4, a: Vec4, b: Vec4, c: Vec4) -> bool {
        let (u, v, w) = Vec4::barycentric(p, a, b, c);
        u >= 0.0 && v >= 0.0 && w >= 0.0 && u <= 1.0 && v <= 1.0 && w <= 1.0
    }

    // ============= Comparison and Validity =============

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
    pub fn approx_eq(&self, other: Vec4) -> bool {
        epsilon_eq(self.x, other.x, 1e-6)
            && epsilon_eq(self.y, other.y, 1e-6)
            && epsilon_eq(self.z, other.z, 1e-6)
            && epsilon_eq(self.w, other.w, 1e-6)
    }

    /// Returns `true` if `self` and `other` are approx equal within `epsilon`.
    ///
    /// This compares each component individually and returns `true` only if both are within epsilon.
    ///
    /// # Parameters
    /// - `other`: The other vector to compare.
    /// - `epsilon`: The maximum allowed difference for each component.
    ///
    /// # Returns
    /// Boolean indicating approx equality.
    pub fn approx_eq_eps(&self, other: Vec4, epsilon: f32) -> bool {
        epsilon_eq(self.x, other.x, epsilon)
            && epsilon_eq(self.y, other.y, epsilon)
            && epsilon_eq(self.z, other.z, epsilon)
            && epsilon_eq(self.w, other.w, epsilon)
    }

    /// Checks whether all components of the vector are finite values.
    ///
    /// A component is considered finite if it is not `NaN` or infinite.
    ///
    /// # Returns
    /// `true` if both `x`, `y`, `z`, and `w` are finite numbers.
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }

    /// Checks whether either component of the vector is `NaN`.
    ///
    /// # Returns
    /// `true` if `x` or `y` or `z` or `w` is `NaN`.
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan() || self.w.is_nan()
    }

    /// Checks whether all components of the vector are approx zero.
    ///
    /// # Returns
    /// `true` if `x`, `y`, `z`, and `w` are approx equal to 0.0.
    pub fn is_zero(self) -> bool {
        epsilon_eq_default(self.x, 0.0)
            && epsilon_eq_default(self.y, 0.0)
            && epsilon_eq_default(self.z, 0.0)
            && epsilon_eq_default(self.w, 0.0)
    }

    /// Checks whether both components of the vector are approx equal to zero within a user-specified tolerance.
    ///
    /// Useful to check for effective zero while avoiding floating point precision issues.
    ///
    /// # Parameters
    /// - `epsilon`: The allowed margin of error.
    ///
    /// # Returns
    /// `true` if `x`, `y`, `z`, and `w` are approx equal to zero within epsilon, `false` otherwise.
    pub fn is_zero_eps(self, epsilon: f32) -> bool {
        epsilon_eq(self.x, 0.0, epsilon)
            && epsilon_eq(self.y, 0.0, epsilon)
            && epsilon_eq(self.z, 0.0, epsilon)
            && epsilon_eq(self.w, 0.0, epsilon)
    }
}

// ============= Operator Overloads =============

use core::f32;
///Addition for Vec4
use std::ops::Add;

/// Adds two vectors together component-wise
///
/// # Parameters
/// - `rhs`: Other vector.
///
/// # Returns
/// `Vec4` which is the sum of `self` and `other`.
impl Add for Vec4 {
    type Output = Vec4;
    fn add(self, rhs: Vec4) -> Self::Output {
        Vec4::new(
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
            self.w + rhs.w,
        )
    }
}

/// Adds each element of the vector with the scalar.
///
/// # Parameters
/// - `scalar`: The float to use.
///
/// # Returns
/// `Vec4` which is the sum `(self.x + scalar, self.y + scalar, self.z + scalar, self.w + scalar)`.
impl Add<f32> for Vec4 {
    type Output = Vec4;
    fn add(self, scalar: f32) -> Self::Output {
        Vec4::new(
            self.x + scalar,
            self.y + scalar,
            self.z + scalar,
            self.w + scalar,
        )
    }
}

/// Adds the float to each element of a given vector.
///
/// # Parameters
/// - `v`: The vector to add.
///
/// # Returns
/// `Vec4` which is the sum `(self + v.x, self + v.y, self + v.z, self + v.w)`.
impl Add<Vec4> for f32 {
    type Output = Vec4;
    fn add(self, v: Vec4) -> Self::Output {
        Vec4::new(self + v.x, self + v.y, self + v.z, self + v.w)
    }
}

///Subtraction for Vec4
use std::ops::Sub;

/// Subtracts `rhs` from `self` component-wise.
///
/// # Parameters
/// - `rhs`: The vector on the right hand side of the calculation.
///
/// # Returns
/// `Vec4` which is the sum `(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z, self.w - rhs.w)`.
impl Sub for Vec4 {
    type Output = Vec4;
    fn sub(self, rhs: Vec4) -> Self::Output {
        Vec4::new(
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z,
            self.w - rhs.w,
        )
    }
}

/// Subtracts `float` from `self` component-wise.
///
/// # Parameters
/// - `scalar`: The float to use.
///
/// # Returns
/// `Vec4` which is the sum `(self.x - scalar, self.y - scalar, self.z - scalar, self.w - scalar)`.
impl Sub<f32> for Vec4 {
    type Output = Vec4;
    fn sub(self, scalar: f32) -> Self::Output {
        Vec4::new(
            self.x - scalar,
            self.y - scalar,
            self.z - scalar,
            self.w - scalar,
        )
    }
}

/// Subtracts `self` from each component of `v`.
///
/// # Parameters
/// - `v`: The vector to use.
///
/// # Returns
/// `Vec4` which is the sum `(scalar - v.x, scalar - v.y, scalar - v.z, scalar - v.w)`.
impl Sub<Vec4> for f32 {
    type Output = Vec4;
    fn sub(self, v: Vec4) -> Self::Output {
        Vec4::new(self - v.x, self - v.y, self - v.z, self - v.w)
    }
}

///Multiplication for Vec4
use std::ops::Mul;

/// Multiplies two vectors together component-wise.
///
/// # Parameters
/// - `rhs`: The vector on the right hand side of the calculation.
///
/// # Returns
/// `Vec4` which is the product of `self * rhs`.
impl Mul for Vec4 {
    type Output = Vec4;
    fn mul(self, rhs: Vec4) -> Self::Output {
        Vec4::new(
            self.x * rhs.x,
            self.y * rhs.y,
            self.z * rhs.z,
            self.w * rhs.w,
        )
    }
}

/// Multiplies each component of a vector by a scalar value.
///
/// # Parameters
/// - `scalar`: The float to use.
///
/// # Returns
/// `Vec4` which is the product of `(self.x * scalar, self.y * scalar, self.z * scalar, self.w * scalar)`.
impl Mul<f32> for Vec4 {
    type Output = Vec4;
    fn mul(self, scalar: f32) -> Self::Output {
        Vec4::new(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
            self.w * scalar,
        )
    }
}

/// Multiplies a float by each component of a vector.
///
/// # Parameters
/// - `vec`: The vector to use.
///
/// # Returns
/// `Vec4` which is the product of `(self * vec.x, self * vec.y, self * vec.z, self * vec.w)`.
impl Mul<Vec4> for f32 {
    type Output = Vec4;
    fn mul(self, vec: Vec4) -> Vec4 {
        Vec4::new(vec.x * self, vec.y * self, vec.z * self, vec.w * self)
    }
}

///Divison for Vec4
use std::ops::Div;

/// Divides vector `self` from `rhs` component-wise.
///
/// # Parameters
/// - `rhs`: The right hand side of the calculation.
///
/// # Returns
/// `Vec4` which is the result of `(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z, self.w / rhs.w)`.
impl Div for Vec4 {
    type Output = Vec4;
    fn div(self, rhs: Vec4) -> Self::Output {
        Vec4::new(
            self.x / rhs.x,
            self.y / rhs.y,
            self.z / rhs.z,
            self.w / rhs.w,
        )
    }
}

/// Divides each component of a vector by `scalar`.
///
/// # Parameters
/// - `scalar`: The float to use.
///
/// # Returns
/// `Vec4` which is the result of `(self.x / scalar, self.y / scalar, self.z / scalar, self.w / scalar)`.
impl Div<f32> for Vec4 {
    type Output = Vec4;
    fn div(self, scalar: f32) -> Self::Output {
        Vec4::new(
            self.x / scalar,
            self.y / scalar,
            self.z / scalar,
            self.w / scalar,
        )
    }
}

/// Divides scalar by each component of a vector.
///
/// # Parameters
/// - `v`: Vector to use.
///
/// # Returns
/// `Vec4` which is the result of `(self / v.x, self / v.y, self / v.z, self / v.w)`.
impl Div<Vec4> for f32 {
    type Output = Vec4;
    fn div(self, vec: Vec4) -> Vec4 {
        Vec4::new(vec.x / self, vec.y / self, vec.z / self, vec.w / self)
    }
}

///Negate Vec4
use std::ops::Neg;

/// Negates all components in a vector to be their negative values.
///
/// This could turn a 1.0 into a -1.0 or vice versa.
///
/// # Returns
/// `Vec4` where all components are negated.
impl Neg for Vec4 {
    type Output = Self;
    fn neg(self) -> Self {
        Vec4::new(-self.x, -self.y, -self.z, -self.w)
    }
}

///Equality check for Vec4
use std::cmp::PartialEq;

/// Checks whether two vectors are exactly equal.
///
/// This doesn't have an epsilon meaning floating point errors may happen.
///
/// # Parameters
/// - `other`: The other vector to compare to.
///
/// # Returns
/// `true` if `self.x == other.x` && `self.y == other.y && self.z == other.z && self.w == other.w`. Otherwise `false`.
impl PartialEq for Vec4 {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z && self.w == other.w
    }
}

use std::ops::{Index, IndexMut};

/// Enables v[index] access.
///
/// # Panics
/// If index >= 4.
impl Index<usize> for Vec4 {
    type Output = f32;

    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Vec4 index out of bounds: {}", index),
        }
    }
}

/// Enables mutable v[index] access.
///
/// # Panics
/// If index >= 4.
impl IndexMut<usize> for Vec4 {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Vec4 index out of bounds: {}", index),
        }
    }
}

/// Method to turn tuples into a vector.
///
/// # Parameters
/// - `t`: The tuple to convert to a vector.
///
/// # Returns
/// `Vec4` which is `(t.0, t.1, t.2, t.3)`.
impl From<(f32, f32, f32, f32)> for Vec4 {
    fn from(t: (f32, f32, f32, f32)) -> Vec4 {
        Vec4::new(t.0, t.1, t.2, t.3)
    }
}

/// Method to turn vectors into tuples.
///
/// # Parameters
/// - `v`: The vector to be converted to a tuple.
///
/// # Returns
/// `tuple` which is `(v.x, v.y, v.z, v.w)`.
impl From<Vec4> for (f32, f32, f32, f32) {
    fn from(v: Vec4) -> (f32, f32, f32, f32) {
        (v.x, v.y, v.z, v.w)
    }
}

/// Method to turn an array into a vector.
///
/// # Parameters
/// - `arr`: Array to be converted.
///
/// # Returns
/// `Vec4` which is `(arr[0], arr[1], arr[2], arr[3])`.
impl From<[f32; 4]> for Vec4 {
    fn from(arr: [f32; 4]) -> Vec4 {
        Vec4::new(arr[0], arr[1], arr[2], arr[3])
    }
}

/// Method to turn a vector into an array.
///
/// # Parameters
/// - `v`: The vector to be converted.
///
/// # Returns
/// `array` which is `[v.x, v.y, v.z, v.w]`.
impl From<Vec4> for [f32; 4] {
    fn from(v: Vec4) -> [f32; 4] {
        [v.x, v.y, v.z, v.w]
    }
}

use std::fmt;

/// Method to correctly display a vector.
///
/// # Parameters
/// - `f`: Formatter to use.
///
/// # Returns
/// `Result` which is a pretty-printed version of the vector.
impl fmt::Display for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vec4({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}
