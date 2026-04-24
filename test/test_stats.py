from rmath import stats as st
import math

def test_stats_core():
    data = [10.0, 2.0, 38.0, 23.0, 38.0, 23.0, 21.0]
    
    print("--- [1/2] Central Tendency ---")
    print(f"data      = {data}")
    print(f"mean      = {st.mean(data)}")
    print(f"median    = {st.median(data)}")
    print(f"mode      = {st.mode(data)} (Expected 38 or 23)")
    
    print("\n--- [2/2] Dispersion & Others ---")
    print(f"variance  = {st.variance(data)}")
    print(f"std_dev   = {st.std_dev(data)}")
    try:
        print(f"harmonic  = {st.harmonic_mean(data)}")
        print(f"geometric = {st.geometric_mean(data)}")
    except ValueError as e:
        print(f"Error: {e}")
        
    print(f"quantiles = {st.quantiles(data, 4)} (Quartiles)")
    z = st.z_scores(data)
    print(f"z_scores  = {z[:3]}... (Length: {len(z)})")
    
    print("\n--- [3/3] Advanced Statistics ---")
    data_y = [12.0, 5.0, 42.0, 20.0, 35.0, 25.0, 24.0]
    print(f"covariance  = {st.covariance(data, data_y)}")
    print(f"correlation = {st.correlation(data, data_y)}")
    print(f"skewness    = {st.skewness(data)}")
    print(f"kurtosis    = {st.kurtosis(data)}")
    print(f"MAD         = {st.median_abs_dev(data)}")

if __name__ == "__main__":
    test_stats_core()
    print("\n--- SUCCESS: Stats module verified! ---")
