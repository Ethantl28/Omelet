# 🥚 Omelet - A Simple Math Library in Rust

Omelet is a lightweight and extensible Rust math library focused on game development. Designed for both clarity and performance, Omelet provides essential vector and matrix math utilities with an emphasis on clean API design, strong documentation, and comprehensive test coverage.

# 
## Features
* 🧮 `Vec2`, `Vec3`, `Vec4` - Fully featured vector types
* 🧊 `Mat2`, `Mat3`, `Mat4` - Matrix types for transformations
* ⭕ `Quat` - Quaternions for 3D rotation
* 📝 Thorough unit tests across all components
* 📃 In-depth documentation with examples (`cargo doc`)
* 📐 Utilities for projection, reflection, barycentric coordinates, SLERP, and more
* 🔄 Operator overloading for intuitive syntax
* ⚙️ (planned) SIMD acceleration for performance-critical operations

#

# 🚀 Getting Started
Add Omelet to your `Cargo.toml`:
```toml
[dependencies]
omelet = {git = "https://github.com/ethantl28/omelet", tag = "v0.1.1"}
```
*Note: *This uses the GitHub version until the crate is published on [crates.io](https://crates.io/crates/omelet)

Once Omelet is added to `crates.io`:
```toml
[dependencies]
omelet = "0.1.1"
```

Import the types you need:
```rust
use omelet::vec::vec2::Vec2;
use omelet::matrices::mat4::Mat4;
```

#
# 🤖 Examples
* Vector addition, dot product, and normalization
```rust
use omelet::vec::Vec2;

fn main() {
    let a = Vec2::new(1.0, 2.0);
    let b = Vec2::new(3.0, 4.0);

    let sum = a + b;
    let dot = a.dot(b);
    let normalized = a.normalize();

    println!("{}, dot: {}, normalized: {}", sum, dot, normalized);
}
```

Output:
```
Vec2(4, 6), dot: 11, normalized: Vec2(0.4472136, 0.8944272)
```
* Vector cross product and reflection
```rust
use omelet::vec::Vec3;

fn main() {
    let a = Vec3::new(1.0, 0.0, 0.0);
    let b = Vec3::new(0.0, 1.0, 0.0);

    let cross = a.cross(b);
    let reflected = a.reflect(b);

    println!("Cross: {}", cross);
    println!("Reflected: {}", reflected);
}
```
Output:
```
Cross: Vec3(0, 0, 1)
Reflected: Vec3(1, 0, 0)
```
* Vector rotation using rotation matrix
```rust
use omelet::matrices::Mat2;

fn main() {
    let rot = Mat2::from_rotation(std::f32::consts::FRAC_2_PI);
    let v = omelet::vec::Vec2::new(1.0, 0.0);
    let rotated = rot * v;

    println!("Rotated vector: {}", rotated);
    println!("Rotation matrix: \n{}", rot);
}
```

Output:
```
Rotated vector: Vec2(0.8041099, 0.59448075)
Rotation matrix:
[[0.8041, -0.5945],
[0.5945, 0.8041]]
```

* Vector rotation using a quaternion
```rust
use omelet::quaternion::Quat;
use omelet::vec::Vec3;

fn main() {
    let axis = Vec3::new(0.0, 1.0, 0.0);
    let angle = std::f32::consts::FRAC_PI_2;
    let rotation = Quat::from_axis_angle(axis, angle);

    let v = Vec3::new(1.0, 0.0, 0.0);
    let rotated = rotation.rotate_vec3(v);

    println!("Rotated Vec3: {}", rotated);
}
```
Output:
```
Rotated Vec3: Vec3(0.000, 0.000, -1.000)
```

* Epsilon comparison
```rust
use omelet::vec::Vec2;
fn main() {
    let a = Vec2::new(1.000001, 2.000001);
    let b = Vec2::new(1.000002, 2.000002);

    assert!(a.approx_eq_eps(b, 1e-5));
    println!("a is approximately equal to b within given epsilon: {}", a.approx_eq_eps(b, 1e-5));
}
```

Output:
```
a is approximately equal to b within given epsilon: true
```

#
# 📃 Documentation
Run locally:
```sh
cargo doc --open
```
Once published, visit: [docs.rs/omelet](https://docs.rs/omelet)
## Vectors
* `Vec2`, `Vec3`, `Vec4` types
* * Extensive unit testing
  * Supports standard operations (addition, subtraction, dot/cross product, normalization, projections, angle calculations, etc.)
  
## Matrices
* `Mat2`, `Mat3`, `Mat4` fully implemented
* Tested against edge cases
* Clean, consistent API
* `Mat4` documentation is ongoing

## Quaternions
* Full quaternion implementation for 3D rotation
* Includes SLERP, normalization, conversion to/from Euler angles
* Heavily tested and documented

## How to run the documentation
To view the full documentation, run:
```
cargo doc --open
```

#
# 📝 Running Tests
Omelet uses Rust's built-in test framework:
```sh
cargo test
```
All modules are tested thoroughly, including edge cases and floating-point comparisons.

#
# 🗺️ Roadmap

* ✅ Matrix functionality parity (`Mat2`, `Mat3`, `Mat4`)
* ✅ Quaternion support with full docs and tests
* 🟨 SIMD acceleration for vector and matrix math
* 🟨 More geometry utilities (plane intersection, AABB, etc.)

#
# 📁 Project Structure
```css
omelet/
├── src/
│   ├── vec/
│   │   ├── mod.rs
│   │   ├── list_of_methods.txt
│   │   ├── vec2.rs   
│   │   ├── vec2_tests.rs
│   │   ├── vec3.rs
│   │   ├── vec3_tests.rs
│   │   ├── vec4.rs
│   │   └── vec4_tests.rs
│   ├── matrices/
│   │   ├── mod.rs
│   │   ├── list_of_methods.txt
│   │   ├── mat2.rs   
│   │   ├── mat2_tests.rs
│   │   ├── mat3.rs
│   │   ├── mat3_tests.rs
│   │   ├── mat4.rs
│   │   └── mat4_tests.rs
│   ├── quat/
│   │   ├── mod.rs
│   │   ├── list_of_methods.txt
│   │   ├── quat.rs   
│   │   └── quat_tests.rs
│   ├── lib.rs
│   └── utils.rs
├── .gitignore
├── Cargo.toml
├── Cargo.lock
└── README.md
```

#
# 🛠️ Contributing
Want to help improve Omelet? Contributions are welcome!
* Please use [pull requests](https://github.com/Ethantl28/Omelet/pulls)
* Code should be formatted using `cargo fmt`
* Ensure tests pass via `cargo tests`
* For major changes, please open an [issue](https://github.com/Ethantl28/Omelet/issues) first
Fork the repo and open a [pull request](https://github.com/Ethantl28/Omelet/pulls) with your improvements.

#
# 💭 Feedback
Have ideas, suggestions, or found a bug? Open an [issue](https://github.com/Ethantl28/Omelet/issues) or start a [discussion](https://github.com/Ethantl28/Omelet/discussions/). 

#
# 📎 License
This project is licensed under the MIT license. See [LICENSE](https://github.com/Ethantl28/Omelet/blob/main/LICENSE) for more information.
