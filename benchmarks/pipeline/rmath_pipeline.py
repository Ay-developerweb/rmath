"""
rmath vs NumPy — Full Data Pipeline Benchmark (rmath side)

Benchmarks a 7-stage financial analytics pipeline on 5 million rows.
Run alongside numpy_pipeline.py to compare results.

Requirements:
    pip install rmath-py psutil

Usage:
    python rmath_pipeline.py
"""
import rmath as rm
import rmath.stats as rs
import time
import psutil
import os

process = psutil.Process(os.getpid())

def mem():
    return process.memory_info().rss / (1024 ** 2)

def log(step, t0, m0):
    print(f"\n--- {step} ---")
    print(f"Time: {time.perf_counter() - t0:.4f}s")
    print(f"Memory delta: {mem() - m0:.2f} MB")


# =====================================
# 1. DATA GENERATION
# =====================================
t0, m0 = time.perf_counter(), mem()

N = 5_000_000

# features
# age: randint(18, 70) -> uniform(0,1)*52 + 18, then floor
# We use Array for generation then convert to Vector for 1D ops
age = rm.Vector((rm.Array.rand_uniform(N) * 52 + 18).floor())

# income/spending: normal distributions using rmath.stats Normal sampler
income = rs.Normal(50000, 15000).sample(N)
spending = rs.Normal(2000, 800).sample(N)

# introduce relationship (noise inlined — no retained variable)
# score = 0.3 * income + 0.7 * spending - 0.2 * age + normal(0, 5000)
score = (income * 0.3) + (spending * 0.7) - (age * 0.2) + rs.Normal(0, 5000).sample(N)

log("Data Generation", t0, m0)


# =====================================
# 2. DATA CLEANING
# =====================================
t0, m0 = time.perf_counter(), mem()

# clip unrealistic values
income = income.clamp(0, 1_000_000)
spending = spending.clamp(0, 1_000_000)

# Single-pass multi-filter (conditions checked ONCE, all 4 vectors filtered together)
age, income, spending, score = rm.Vector.multi_filter_where(
    [age, income, spending, score],
    [(income, "lt", 150000), (spending, "lt", 10000)]
)

log("Cleaning", t0, m0)


# =====================================
# 3. FEATURE ENGINEERING
# =====================================
t0, m0 = time.perf_counter(), mem()

# ratio + interaction features
spend_ratio = spending / (income + 1)
wealth_index = (income * 0.6) + (spending * 0.4)

log("Feature Engineering", t0, m0)


# =====================================
# 4. DESCRIPTIVE STATS
# =====================================
t0, m0 = time.perf_counter(), mem()

def stats(name, v):
    print(f"{name}: mean={v.mean():.2f}, std={v.std_dev():.2f}, min={v.min():.2f}, max={v.max():.2f}")

stats("income", income)
stats("spending", spending)
stats("score", score)

log("Descriptive Stats", t0, m0)


# =====================================
# 5. CORRELATION ANALYSIS
# =====================================
t0, m0 = time.perf_counter(), mem()

# Use rmath.stats.correlation for Pearson r
corr_income = rs.correlation(income, score)
corr_spending = rs.correlation(spending, score)
corr_age = rs.correlation(age, score)

print("\nCorrelations:")
print(f"income vs score: {corr_income:.4f}")
print(f"spending vs score: {corr_spending:.4f}")
print(f"age vs score: {corr_age:.4f}")

log("Correlation", t0, m0)


# =====================================
# 6. GROUPED ANALYSIS (SEGMENTATION)
# =====================================
t0, m0 = time.perf_counter(), mem()

# segment by age group using filter_where (zero mask allocations)
# Group 0: age <= 25
# Group 1: 25 < age <= 35
# Group 2: 35 < age <= 50
# Group 3: age > 50
group_conditions = [
    [(age, "le", 25.0)],
    [(age, "gt", 25.0), (age, "le", 35.0)],
    [(age, "gt", 35.0), (age, "le", 50.0)],
    [(age, "gt", 50.0)],
]

for group, conds in enumerate(group_conditions):
    g_income = income.filter_where(conds)
    g_spending = spending.filter_where(conds)
    g_score = score.filter_where(conds)
    if g_income.size > 0:
        print(f"\nGroup {group}:")
        print(f"avg income: {g_income.mean():.2f}")
        print(f"avg spending: {g_spending.mean():.2f}")
        print(f"avg score: {g_score.mean():.2f}")

log("Segmentation", t0, m0)


# =====================================
# 7. SIMPLE LINEAR SIGNAL
# =====================================
t0, m0 = time.perf_counter(), mem()

# High-performance linear regression from rmath.stats
res = rs.linear_regression(income, score)

print(f"\nLinear relationship (income -> score): {res['slope']:.4f}")

log("Linear Signal", t0, m0)
