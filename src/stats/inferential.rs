use pyo3::prelude::*;
use rayon::prelude::*;
use crate::vector::Vector;

/// Student's T-Distribution CDF approximation
fn t_cdf_approx(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    // Rough approximation for p-value logic
    if t > 0.0 {
        1.0 - 0.5 * x.powf(df / 2.0)
    } else {
        0.5 * x.powf(df / 2.0)
    }
}

/// Calculate the Pearson correlation coefficient between two variables.
///
/// Returns a value between -1 and 1. Values close to 1 indicate a strong
/// positive linear relationship; values close to -1 indicate a strong
/// negative linear relationship.
///
/// Examples:
///     >>> from rmath.stats import correlation
///     >>> correlation([1, 2, 3], [2, 4, 6])
///     1.0
#[pyfunction]
pub fn correlation(x: Bound<'_, PyAny>, y: Bound<'_, PyAny>) -> PyResult<f64> {
    let vx: PyRef<Vector> = x.extract()?;
    let vy: PyRef<Vector> = y.extract()?;
    correlation_internal(&vx, &vy)
}

fn correlation_internal(vx: &Vector, vy: &Vector) -> PyResult<f64> {
    vx.with_slice(|sx| vy.with_slice(|sy| {
        let n = sx.len();
        if n != sy.len() || n < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("Size mismatch or n < 2"));
        }
        let mx = sx.par_iter().sum::<f64>() / n as f64;
        let my = sy.par_iter().sum::<f64>() / n as f64;
        let (num, den_x, den_y) = sx.par_iter().zip(sy.par_iter())
            .map(|(&xi, &yi)| {
                let dx = xi - mx;
                let dy = yi - my;
                (dx * dy, dx * dx, dy * dy)
            })
            .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        let den = (den_x * den_y).sqrt();
        if den == 0.0 { Ok(0.0) } else { Ok(num / den) }
    }))
}

/// Calculate the sample covariance between two variables.
#[pyfunction]
pub fn covariance(x: Bound<'_, PyAny>, y: Bound<'_, PyAny>) -> PyResult<f64> {
    let vx: PyRef<Vector> = x.extract()?;
    let vy: PyRef<Vector> = y.extract()?;
    vx.with_slice(|sx| vy.with_slice(|sy| {
        let n = sx.len();
        if n != sy.len() || n < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("Size mismatch or n < 2"));
        }
        let mx = sx.par_iter().sum::<f64>() / n as f64;
        let my = sy.par_iter().sum::<f64>() / n as f64;
        let sum_prod = sx.par_iter().zip(sy.par_iter())
            .map(|(&xi, &yi)| (xi - mx) * (yi - my))
            .sum::<f64>();
        Ok(sum_prod / (n - 1) as f64)
    }))
}

/// Independent T-Test (Unequal Variances / Welch's T-test)
/// Returns (T-Statistic, P-Value)
/// Perform an independent Student's T-test (Welch's T-test).
///
/// Returns a tuple of (T-Statistic, P-Value). This test assumes
/// unequal variances between the two groups.
///
/// Examples:
///     >>> from rmath.stats import t_test_independent
///     >>> t, p = t_test_independent([1, 2, 3], [4, 5, 6])
#[pyfunction]
pub fn t_test_independent(a: Bound<'_, PyAny>, b: Bound<'_, PyAny>) -> PyResult<(f64, f64)> {
    let va: PyRef<Vector> = a.extract()?;
    let vb: PyRef<Vector> = b.extract()?;
    
    let (m1, v1, n1) = (va.mean(), va.variance(), va.len_internal() as f64);
    let (m2, v2, n2) = (vb.mean(), vb.variance(), vb.len_internal() as f64);
    
    let t_stat = (m1 - m2) / (v1/n1 + v2/n2).sqrt();
    
    // Degrees of freedom (Welch–Satterthwaite)
    let df_num = (v1/n1 + v2/n2).powi(2);
    let df_den = (v1/n1).powi(2)/(n1-1.0) + (v2/n2).powi(2)/(n2-1.0);
    let df = df_num / df_den;
    
    // P-value approximation
    let p_val = 2.0 * (1.0 - t_cdf_approx(t_stat.abs(), df));
    
    Ok((t_stat, p_val))
}

/// Calculate the Spearman rank correlation coefficient.
///
/// This is a non-parametric measure of rank correlation (statistical dependence
/// between the rankings of two variables).
#[pyfunction]
pub fn spearman_correlation(x: Bound<'_, PyAny>, y: Bound<'_, PyAny>) -> PyResult<f64> {
    let vx: PyRef<Vector> = x.extract()?;
    let vy: PyRef<Vector> = y.extract()?;
    let n = vx.len_internal();
    if n != vy.len_internal() || n < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("Size mismatch or n < 2"));
    }
    fn get_ranks(v: &Vector) -> Vector {
        let n = v.len_internal();
        let idx = v.argsort();
        let mut ranks = vec![0.0; n];
        v.with_slice(|s| {
            let mut i = 0;
            while i < n {
                let mut j = i + 1;
                while j < n && s[idx[j] as usize] == s[idx[i] as usize] { j += 1; }
                let mean_rank = (i + j - 1) as f64 / 2.0;
                for k in i..j { ranks[idx[k] as usize] = mean_rank; }
                i = j;
            }
        });
        Vector::new(ranks)
    }
    let rx = get_ranks(&vx);
    let ry = get_ranks(&vy);
    correlation_internal(&rx, &ry)
}

/// Perform a One-Way analysis of variance (ANOVA).
///
/// Returns a tuple of (F-Statistic, Degrees of Freedom Between, Degrees of Freedom Within).
#[pyfunction]
pub fn anova_oneway(groups: Vec<Bound<'_, PyAny>>) -> PyResult<(f64, f64, f64)> {
    let mut n_total = 0.0;
    let mut sum_total = 0.0;
    let mut group_stats = Vec::new();
    for g in groups {
        let v: PyRef<Vector> = g.extract()?;
        let (m, m2, n) = v.welford();
        if n == 0 { continue; }
        let nf = n as f64;
        n_total += nf;
        sum_total += m * nf;
        group_stats.push((nf, m, m2));
    }
    if group_stats.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("Need at least 2 groups for ANOVA"));
    }
    let overall_mean = sum_total / n_total;
    let ss_between: f64 = group_stats.iter()
        .map(|(ni, mi, _)| ni * (mi - overall_mean).powi(2))
        .sum();
    let ss_within: f64 = group_stats.iter().map(|(_, _, ssi)| ssi).sum();
    let df_between = (group_stats.len() - 1) as f64;
    let df_within = n_total - group_stats.len() as f64;
    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;
    let f_stat = ms_between / ms_within;
    Ok((f_stat, df_between, df_within))
}
