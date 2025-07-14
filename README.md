# 🥚 Omelet - A Simple Math Library in Rust

Omelet is a lightweight and extensible Rust math library focused on game development. Designed for both clarity and performance, Omelet provides essential vector and matrix math utilities with an emphasis on clean API design, strong documentation, and comprehensive test coverage.

# 
## Features
* 🧮 `Vec2`, `Vec3`, `Vec4` - Fully featured vector types
* 🧊 `Mat2`, `Mat3`, `Mat4` - Matrix types for transformations (WIP)
* 📝 Extensive unit tests for all vector types, matrix documentation limited 
* 📃 Comprehensive API documentation for vectors via `cargo doc`
* 📐 Math utilities: projection, reflection, barycentric coordinates, slerp, etc
* 🔄 Operator overloading for natural syntax
* ⚙️ Future goals: SIMD acceleration, improved matrix support, transform types

#

# 🚀 Getting Started
Add Omelet to your `Cargo.toml`:
```
[dependencies]
omelet = {path = "path_to_omelet"}
```

Once Omelet is added to `crates.io`:
```
[dependencies]
omelet = 0.1.0-alpha
```

Import the types you need:
```
use omelet::vec2::Vec2;
use omelet::mat4::Mat4;
```

#
# 📃 Documentation
## Vectors
All vector types (`Vec2`, `Vec3`, `Vec4`) are:
* Well tested with extensive unit test coverage
* Thoroughly documented with usage examples
* Support most common vector operations (addition, dot/cross, normalization, rotation, projection, etc)
To view the full documentation, run:
```
cargo doc --open
```
## Matrices
Matrix types `Mat2`, `Mat3`, and `Mat4` are currently:
* Implemented with basic functionality
* Missing full documentation and unit test coverage
Expect significant updates and improvements in upcoming versions. 

#
# 📝 Testing
Omelet uses Rust's built-in test framework. To run the tests:
```
cargo test
```
All vector functions are covered by rigorous tests, including edge cases and floating-point epsilon comparisons.

#
# 🗺️ Roadmap

* Additional functionality for all matrix types
* SIMD acceleration for vectors
* SIMD acceleration for matrices

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
│   │   ├── mat2.rs   
│   │   ├── mat2_tests.rs
│   │   ├── mat3.rs
│   │   ├── mat3_tests.rs
│   │   ├── mat4.rs
│   │   └── mat4_tests.rs
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
