pub mod mat2;
pub mod mat3;
pub mod mat4;

//Re export
pub use mat2::Mat2;
pub use mat3::Mat3;
pub use mat4::Mat4;

//Tests
#[cfg(test)]
mod mat2_tests;

#[cfg(test)]
mod mat3_tests;

#[cfg(test)]
mod mat4_tests;
