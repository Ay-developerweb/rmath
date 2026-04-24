"""
rmath vs NumPy — Full Data Pipeline Benchmark (NumPy side)

Benchmarks a 7-stage financial analytics pipeline on 5 million rows.
Run alongside rmath_pipeline.py to compare results.

Requirements:
    pip install numpy psutil

Usage:
    python numpy_pipeline.py
"""
import numpy as np
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
age = np.random.randint(18, 70, N)
income = np.random.normal(50000, 15000, N)
spending = np.random.normal(2000, 800, N)

# introduce relationship
score = 0.3 * income + 0.7 * spending - 0.2 * age + np.random.normal(0, 5000, N)

log("Data Generation", t0, m0)


# =====================================
# 2. DATA CLEANING
# =====================================
t0, m0 = time.perf_counter(), mem()

# clip unrealistic values
income = np.clip(income, 0, None)
spending = np.clip(spending, 0, None)

# remove outliers (z-score approx)
mask = (income < 150000) & (spending < 10000)

age = age[mask]
income = income[mask]
spending = spending[mask]
score = score[mask]

log("Cleaning", t0, m0)


# =====================================
# 3. FEATURE ENGINEERING
# =====================================
t0, m0 = time.perf_counter(), mem()

# ratio + interaction features
spend_ratio = spending / (income + 1)
wealth_index = income * 0.6 + spending * 0.4

log("Feature Engineering", t0, m0)


# =====================================
# 4. DESCRIPTIVE STATS
# =====================================
t0, m0 = time.perf_counter(), mem()

def stats(name, arr):
    print(f"{name}: mean={np.mean(arr):.2f}, std={np.std(arr):.2f}, min={np.min(arr):.2f}, max={np.max(arr):.2f}")

stats("income", income)
stats("spending", spending)
stats("score", score)

log("Descriptive Stats", t0, m0)


# =====================================
# 5. CORRELATION ANALYSIS
# =====================================
t0, m0 = time.perf_counter(), mem()

corr_income = np.corrcoef(income, score)[0, 1]
corr_spending = np.corrcoef(spending, score)[0, 1]
corr_age = np.corrcoef(age, score)[0, 1]

print("\nCorrelations:")
print("income vs score:", corr_income)
print("spending vs score:", corr_spending)
print("age vs score:", corr_age)

log("Correlation", t0, m0)


# =====================================
# 6. GROUPED ANALYSIS (SEGMENTATION)
# =====================================
t0, m0 = time.perf_counter(), mem()

# segment by age group
bins = np.digitize(age, [25, 35, 50])

for group in range(4):
    mask = bins == group
    if np.sum(mask) > 0:
        print(f"\nGroup {group}:")
        print("avg income:", np.mean(income[mask]))
        print("avg spending:", np.mean(spending[mask]))
        print("avg score:", np.mean(score[mask]))

log("Segmentation", t0, m0)


# =====================================
# 7. SIMPLE LINEAR SIGNAL (MANUAL)
# =====================================
t0, m0 = time.perf_counter(), mem()

# slope estimate (income -> score)
cov = np.mean((income - np.mean(income)) * (score - np.mean(score)))
var = np.var(income)

beta = cov / var

print("\nLinear relationship (income -> score):", beta)

log("Linear Signal", t0, m0)
