import rmath

def demonstrate_rmath_vector():
    # 1. Initialize data
    raw_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    print(f"Original List: {raw_data}")

    # 2. Create the Vector object (Now available at the top level!)
    v = rmath.Vector(raw_data)
    print(f"rmath.Vector object created with length: {v.len()}")

    # 3. Arithmetic Transformation (Chaining)
    # Note: Chaining is where the real speed comes from
    v_transformed = v.add_scalar(10.0).mul_scalar(0.5).sin()
    print(f"Result (Chained: (v + 10) * 0.5 -> sin): {v_transformed.to_list()}")

    # 4. Individual Operations
    print("\n--- [1/3] Individual Operations ---")
    print(f"Sqrt:  {v.sqrt().to_list()}")
    print(f"Log:   {v.log().to_list()}")
    print(f"Abs:   {v.abs().to_list()}")
    print(f"Ceil:  {v.add_scalar(0.5).ceil().to_list()}") # Rounds 1.5, 2.5... up

    # 5. Trigonometry
    print("\n--- [2/3] Trigonometry ---")
    print(f"Sin:   {v.sin().to_list()}")
    print(f"Cos:   {v.cos().to_list()}")
    print(f"Tan:   {v.tan().to_list()}")

    # 6. Statistical Reductions (Scalar Results)
    print("\n--- [3/3] Statistical Reductions ---")
    print(f"Sum:      {v.sum()}")
    print(f"Mean:     {v.mean()}")
    print(f"Variance: {v.variance()}")
    print(f"Std Dev:  {v.std_dev()}")

    # 7. Convert back to list when done
    final_list = v_transformed.to_list()
    print(f"\nFinal Result as Python list: {final_list}")

if __name__ == "__main__":
    demonstrate_rmath_vector()
