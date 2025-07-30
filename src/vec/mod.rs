pub mod vec2;
pub mod vec3;
pub mod vec4;

//Re export
pub use vec2::Vec2;
pub use vec3::Vec3;
pub use vec4::Vec4;

//Tests
#[cfg(test)]
mod vec2_tests;

#[cfg(test)]
mod vec3_tests;

#[cfg(test)]
mod vec4_tests;
