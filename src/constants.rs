use pyo3::prelude::*;

/// Registers the constants submodule (rmath.constants).
pub fn register_constants(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__doc__",
        "Mathematical and physical constants.\n\n\
        Provides high-precision constants including PI, E, PHI, TAU, and machine \
        epsilons, bypassing Python's `math` module overhead.",
    )?;

    // --- Fundamental Mathematical Constants ---
    m.add("PI",       std::f64::consts::PI)?;         // 3.14159…
    m.add("TAU",      std::f64::consts::TAU)?;        // 2π = 6.28318…
    m.add("E",        std::f64::consts::E)?;           // 2.71828…
    m.add("PHI",      1.6180339887498948482_f64)?;    // Golden ratio
    m.add("SQRT_2",   std::f64::consts::SQRT_2)?;     // √2
    m.add("SQRT_3",   1.7320508075688772935_f64)?;    // √3
    m.add("SQRT_5",   2.2360679774997896964_f64)?;    // √5
    m.add("SQRT_1_2", std::f64::consts::FRAC_1_SQRT_2)?; // 1/√2

    // --- Logarithm Constants ---
    m.add("LN_2",    std::f64::consts::LN_2)?;        // ln(2)
    m.add("LN_10",   std::f64::consts::LN_10)?;       // ln(10)
    m.add("LOG2_E",  std::f64::consts::LOG2_E)?;      // log₂(e)
    m.add("LOG10_E", std::f64::consts::LOG10_E)?;     // log₁₀(e)

    // --- Machine-Precision Constants ---
    m.add("EPSILON_F32", f32::EPSILON as f64)?;       // ~1.19e-07 (float32 epsilon)
    m.add("EPSILON_F64", f64::EPSILON)?;              // ~2.22e-16 (float64 epsilon)
    m.add("MAX_F32",     f32::MAX as f64)?;           // Maximum f32
    m.add("MAX_F64",     f64::MAX)?;                  // Maximum f64
    m.add("MIN_POSITIVE_F64", f64::MIN_POSITIVE)?;   // Smallest positive f64

    // --- Special Values ---
    m.add("INF",     f64::INFINITY)?;
    m.add("NEG_INF", f64::NEG_INFINITY)?;
    m.add("NAN",     f64::NAN)?;

    Ok(())
}
