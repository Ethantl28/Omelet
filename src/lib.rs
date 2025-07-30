//! 🥚 Omelet - A Simple Math Library for Games and Graphics
//!
//!  Omelet provides vector, matrix, and quaternion types with clean APIs and full test coverage.
//! It's built for game devs, graphics programmers, and anyone needing fast, clear math tools.
//!
//! # Features
//! - Vec2, Vec3, Vec4 types
//! - Mat2, Mat3, Mat4 with common transforms
//! - Quat with SLERP, Euler conversion, and more
//! - Operator overloading, epsilon-aware comparisons
//! - Extensive testing and documentation

pub mod matrices;
pub mod quaternion;
pub mod utils;
pub mod vec;

pub use matrices::{mat2, mat3, mat4};
pub use quaternion::Quat;
pub use utils::epsilon_eq;
pub use vec::{vec2, vec3, vec4};
