use pyo3::prelude::*;
use crate::vector::core::Vector;

/// A Quaternion for 3D rotations: q = w + xi + yj + zk.
///
/// Quaternions provide a way to represent and compose 3D rotations without
/// the risk of gimbal lock.
#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[pymethods]
impl Quaternion {
    /// Create a new Quaternion: w + xi + yj + zk.
    #[new]
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Quaternion { w, x, y, z }
    }

    /// Calculate the L2 norm of the quaternion.
    pub fn norm(&self) -> f64 {
        (self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z).sqrt()
    }

    /// Return a normalized (unit) version of this quaternion.
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n == 0.0 { *self }
        else { Quaternion { w: self.w/n, x: self.x/n, y: self.y/n, z: self.z/n } }
    }

    /// Rotate a 3D vector using this quaternion.
    ///
    /// The rotation is performed as `v' = q * v * q⁻¹`.
    pub fn rotate_vector(&self, v: &Vector) -> PyResult<Vector> {
        v.with_slice(|s| {
            if s.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err("Can only rotate 3D vectors"));
            }
            // q * v * q_inv
            let q_vec = Quaternion { w: 0.0, x: s[0], y: s[1], z: s[2] };
            let q_inv = Quaternion { w: self.w, x: -self.x, y: -self.y, z: -self.z };
            
            let res = self.mul_internal(&q_vec).mul_internal(&q_inv);
            Ok(Vector::new(vec![res.x, res.y, res.z]))
        })
    }

    pub fn __mul__(&self, other: &Quaternion) -> Self {
        self.mul_internal(other)
    }

    pub fn __repr__(&self) -> String {
        format!("Quaternion(w={}, x={}, y={}, z={})", self.w, self.x, self.y, self.z)
    }
}

impl Quaternion {
    fn mul_internal(&self, other: &Quaternion) -> Self {
        Quaternion {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }
}
