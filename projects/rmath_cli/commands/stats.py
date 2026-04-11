from rmath.vector import Vector
from rmath.stats import describe, skewness, kurtosis
from ..utils.formatter import console, print_stats_table, progress_bar
import numpy as np

def run_stats_analysis(data_source, n_points=100000):
    with progress_bar() as progress:
        t1 = progress.add_task("[cyan]Generating data...", total=100)
        
        if data_source == "random":
            v = Vector.randn(n_points)
        elif data_source == "linear":
            v = Vector.linspace(0, 100, n_points)
        else:
            # Placeholder for file loading in future
            v = Vector.zeros(n_points)
            
        progress.update(t1, advance=50)
        
        t2 = progress.add_task("[magenta]Computing moments...", total=100)
        stats = describe(v)
        sk = skewness(v)
        kt = kurtosis(v)
        progress.update(t2, advance=100)

    res = {
        "Count": len(v),
        "Mean": stats["mean"],
        "Variance": stats["variance"],
        "Std Dev": stats["std_dev"],
        "Min": stats["min"],
        "Max": stats["max"],
        "Skewness": sk,
        "Kurtosis": kt
    }
    
    print_stats_table(res, title=f"Descriptive Statistics (N={n_points})")
    
    if abs(sk) < 0.1 and abs(kt - 3.0) < 0.2:
        console.print("[success]Data appears to follow a Normal Distribution.[/success]")
    else:
        console.print("[warning]Data shows non-normal characteristics.[/warning]")
