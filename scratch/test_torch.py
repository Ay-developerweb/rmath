# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import rmath as rm

print("Testing PyTorch interop...")
try:
    import torch
    t = torch.tensor([[1.0, 2.0]])
    print(f"  1. Original torch: {t}")
    
    a = rm.Array.from_torch(t)
    print(f"  2. from_torch OK: {a}")
    
    # The to_torch method goes: Array -> numpy -> torch.from_numpy
    # Let's test step by step
    np_arr = a.to_numpy()
    print(f"  3. to_numpy OK: {np_arr}, type: {type(np_arr)}")
    
    # Manual torch conversion (workaround)
    import numpy as np
    t2 = torch.from_numpy(np.array(np_arr))
    print(f"  4. Manual roundtrip OK: {t2}")
    assert torch.allclose(t, t2)
    print("  PASS: Manual workaround works")
    
    # Now test the built-in to_torch
    try:
        t3 = a.to_torch()
        print(f"  5. to_torch OK: {t3}")
    except Exception as e:
        print(f"  5. to_torch FAIL: {e}")
        print("     (Known issue: numpy crate version mismatch with torch.from_numpy)")
        
except ImportError:
    print("  PyTorch not installed, skipping")
