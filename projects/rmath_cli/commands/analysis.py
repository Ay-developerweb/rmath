import polars as pl
from rmath.vector import Vector
from rmath.stats import describe, t_test_independent
from ..utils.formatter import console, print_stats_table, progress_bar, Panel
import time
import os

def run_data_analysis(file_path, col_a, col_b=None):
    if not os.path.exists(file_path):
        console.print(f"[error]File not found: {file_path}[/error]")
        return

    with progress_bar() as progress:
        t1 = progress.add_task(f"[cyan]Loading {file_path} via Polars...", total=100)
        df = pl.read_csv(file_path)
        progress.update(t1, advance=100)

        # Extraction and conversion to Rmath Vector
        if col_a not in df.columns:
            console.print(f"[error]Column '{col_a}' not found.[/error]")
            return
            
        t2 = progress.add_task(f"[magenta]Analyzing {col_a}...", total=100)
        # Shift data to Rmath
        v_a = Vector(df[col_a].to_list())
        stats_a = describe(v_a)
        progress.update(t2, advance=100)

        if col_b and col_b in df.columns:
            t3 = progress.add_task(f"[magenta]Comparing with {col_b}...", total=100)
            v_b = Vector(df[col_b].to_list())
            # Independent T-Test using Rmath's inferential engine
            t_stat, p_val = t_test_independent(v_a, v_b)
            progress.update(t3, advance=100)
        else:
            t_stat, p_val = None, None

    # UI Output
    console.print(Panel(f"Loaded [bold white]{len(df)}[/bold white] rows from dataset.", title="Data Pipeline", border_style="blue"))
    
    print_stats_table(stats_a, title=f"Statistical Profile: {col_a}")
    
    if t_stat is not None:
        table = print_stats_table({
            "T-Statistic": t_stat,
            "P-Value": p_val,
            "Null Hypothesis": "Means are equal",
            "Status": "[success]REJECTED[/success]" if p_val < 0.05 else "[warning]FAILED TO REJECT[/warning]"
        }, title=f"Inferential Analysis: {col_a} vs {col_b}")
