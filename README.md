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
```
[dependencies]
omelet = {git = "https://github.com/ethantl28/omelet", tag = "v0.1.0-alpha"}
```
*Note: *This uses the GitHub version until the crate is published on [crates.io](https://crates.io/crates/omelet)

Once Omelet is added to `crates.io`:
```
[dependencies]
omelet = 0.1.0-alpha
```

Import the types you need:
```
use omelet::vec::vec2::Vec2;
use omelet::matrices::mat4::Mat4;
```

#
# 📃 Documentation
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
* Heavily testes and documented

## How to run the documentation
To view the full documentation, run:
```
cargo doc --open
```

#
# 📝 Running Tests
Omelet uses Rust's built-in test framework:
```
cargo test
```
All modules are tested thoroughly, including edge cases and floating-point comparisons.

#
# 📔 Building Documentation
Gnertate and open documentation locally with:
```
cargo doc --open
```

#
# 🗺️ Roadmap

* ✅ Matrix functionality parity (`Mat2`, `Mat3`, `Mat4`)
* ✅ Quaternion support with full docs and tests
* 🟨 SIMD acceleration for vector and matrix math
* 🟨 More geometry utilities (plane intersection, AABB, etc.)

#
# 📁 Project Structure
```
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
