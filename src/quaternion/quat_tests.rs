use crate::{
    vec::{Vec3, Vec4},
    Quat,
};
use approx::AbsDiffEq;
use std::f32::consts::{FRAC_PI_2, PI};

// Helper for comparing quaternions, accounting for the fact that q and -q represent the same rotation.
fn quat_approx_eq(a: Quat, b: Quat) -> bool {
    a.abs_diff_eq(&b, 1e-5) || a.abs_diff_eq(&-b, 1e-5)
}

// Helper for comparing vectors.
fn vec3_approx_eq(a: Vec3, b: Vec3) -> bool {
    a.abs_diff_eq(&b, 1e-5)
}

#[test]
fn test_euler_angles_conversion() {
    let roll = 0.1;
    let pitch = 0.2;
    let yaw = 0.3;

    let q = Quat::from_euler_angles(roll, pitch, yaw);
    let (r2, p2, y2) = q.to_euler_angles();

    let q2 = Quat::from_euler_angles(r2, p2, y2);
    let dot = q.dot(q2);
    assert!(
        (1.0 - dot.abs()) < 1e-5,
        "Quats differ: q={:?}, q2={:?}, dot={}",
        q,
        q2,
        dot
    );
}

#[test]
fn test_swing_twist() {
    let twist_axis = Vec3::new(0.0, 1.0, 0.0).normalize();
    let rot = Quat::from_euler_angles(0.5, 0.8, 0.3);

    let (swing, twist) = rot.to_swing_twist(twist_axis);
    assert!(quat_approx_eq(rot, swing * twist));
    let (twist_rot_axis, twist_angle) = twist.to_axis_angle();
    if twist_angle.abs() > 1e-6 {
        assert!(
            vec3_approx_eq(twist_rot_axis, twist_axis)
                || vec3_approx_eq(twist_rot_axis, -twist_axis)
        );
    }

    // 3. The swing component's rotation axis must be perpendicular to the twist axis.
    let (swing_rot_axis, swing_angle) = swing.to_axis_angle();
    if swing_angle.abs() > 1e-6 {
        assert!(swing_rot_axis.dot(twist_axis).abs() < 1e-6);
    }
}

#[test]
fn test_construction_and_constants() {
    let q = Quat::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(q.x, 1.0);
    assert_eq!(q.y, 2.0);
    assert_eq!(q.z, 3.0);
    assert_eq!(q.w, 4.0);

    assert_eq!(Quat::IDENTITY, Quat::new(0.0, 0.0, 0.0, 1.0));
    assert_eq!(Quat::ZERO, Quat::new(0.0, 0.0, 0.0, 0.0));
    assert!(Quat::NAN.is_nan());
}

#[test]
fn test_from_axis_angle() {
    let axis = Vec3::new(0.0, 1.0, 0.0);
    let angle = FRAC_PI_2; // 90 degrees
    let q = Quat::from_axis_angle(axis, angle);

    let expected = Quat::new(0.0, (angle / 2.0).sin(), 0.0, (angle / 2.0).cos());
    assert!(quat_approx_eq(q, expected));

    // Test zero angle
    let q_zero = Quat::from_axis_angle(axis, 0.0);
    assert!(quat_approx_eq(q_zero, Quat::IDENTITY));
}

#[test]
fn test_from_rotation_axes() {
    let angle = FRAC_PI_2;
    let qx = Quat::from_rotation_x(angle);
    let qy = Quat::from_rotation_y(angle);
    let qz = Quat::from_rotation_z(angle);

    assert!(quat_approx_eq(
        qx,
        Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), angle)
    ));
    assert!(quat_approx_eq(
        qy,
        Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), angle)
    ));
    assert!(quat_approx_eq(
        qz,
        Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), angle)
    ));
}

#[test]
fn test_vec4_conversion() {
    let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let q = Quat::from_vec4(v);
    assert_eq!(q.x, 1.0);
    assert_eq!(q.y, 2.0);
    assert_eq!(q.z, 3.0);
    assert_eq!(q.w, 4.0);

    let v_back = q.to_vec4();
    assert_eq!(v, v_back);
}

#[test]
fn test_vector_and_scalar_parts() {
    let q = Quat::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(q.vector_part(), Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(q.scalar_part(), 4.0);
}

#[test]
fn test_length_and_normalization() {
    let q = Quat::new(1.0, 2.0, 3.0, 4.0);
    assert!((q.length() - (30.0_f32).sqrt()).abs() < 1e-6);
    assert!((q.squared_length() - 30.0).abs() < 1e-6);

    let normalized = q.normalize();
    assert!((normalized.length() - 1.0).abs() < 1e-6);
    assert!(normalized.is_normalized());

    // Test normalization of zero quaternion
    let zero_norm = Quat::ZERO.normalize();
    assert_eq!(zero_norm, Quat::IDENTITY);

    // Test try_normalize
    assert!(q.try_normalize().is_some());
    assert!(Quat::ZERO.try_normalize().is_none());
}

#[test]
fn test_inverse_and_conjugate() {
    let q = Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), 0.5);

    // For a unit quaternion, inverse is the conjugate
    let inv = q.inverse();
    let conj = q.conjugate();
    assert!(quat_approx_eq(inv, conj));

    // q * q.inverse() should be identity
    let identity = q * inv;
    assert!(quat_approx_eq(identity, Quat::IDENTITY));
}

#[test]
fn test_vector_rotation() {
    let angle = FRAC_PI_2; // 90 degrees around Y
    let q = Quat::from_rotation_y(angle);

    let v = Vec3::new(1.0, 0.0, 0.0);
    let rotated_v = q.rotate_vec3(v);
    let expected_v = Vec3::new(0.0, 0.0, -1.0);

    assert!(vec3_approx_eq(rotated_v, expected_v));

    // Test operator overload
    let rotated_v_op = q * v;
    assert!(vec3_approx_eq(rotated_v_op, expected_v));
}

#[test]
fn test_from_to_rotation() {
    let from = Vec3::new(1.0, 0.0, 0.0);
    let to = Vec3::new(0.0, 1.0, 0.0);
    let q = Quat::from_to_rotation(from, to);
    let rotated = q * from;
    assert!(vec3_approx_eq(rotated, to));

    // Test opposite vectors
    let from_opp = Vec3::new(1.0, 0.0, 0.0);
    let to_opp = Vec3::new(-1.0, 0.0, 0.0);
    let q_opp = Quat::from_to_rotation(from_opp, to_opp);
    let rotated_opp = q_opp * from_opp;
    assert!(vec3_approx_eq(rotated_opp, to_opp));

    // Test identical vectors
    let q_ident = Quat::from_to_rotation(from, from);
    assert!(quat_approx_eq(q_ident, Quat::IDENTITY));
}

#[test]
fn test_slerp_and_nlerp() {
    let q1 = Quat::IDENTITY;
    let q2 = Quat::from_rotation_y(FRAC_PI_2);

    // Slerp
    let slerp_half = q1.slerp(q2, 0.5);
    let expected = Quat::from_rotation_y(FRAC_PI_2 / 2.0);
    assert!(quat_approx_eq(slerp_half, expected));

    // Nlerp should be normalized and point in a similar direction
    let nlerp_half = q1.nlerp(q2, 0.5);
    assert!(nlerp_half.is_normalized());
    assert!(nlerp_half.dot(expected) > 0.99); // Check that it's close to the slerp result

    // Test slerp with quaternions that need the shorter path
    let q3 = Quat::from_rotation_y(PI * 0.1);
    let q4 = -Quat::from_rotation_y(PI * 0.3); // opposite hemisphere
    let slerp_short = q3.slerp(q4, 0.5);
    let expected_short = Quat::from_rotation_y(PI * 0.2);
    assert!(quat_approx_eq(slerp_short, expected_short));
}
