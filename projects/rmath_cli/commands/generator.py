import polars as pl
from rmath.vector import Vector
from ..utils.formatter import console, progress_bar
import time

def run_data_generation(out_file, n_rows=1_000_000):
    console.print(f"[info]Synthesizing [bold white]{n_rows}[/bold white] rows of high-fidelity sensor data...[/info]")
    
    with progress_bar() as progress:
        # Time Axis
        t1 = progress.add_task("[cyan]Generating Time Matrix...", total=100)
        times = Vector.linspace(0, 1000, n_rows)
        progress.update(t1, advance=100)
        
        # Sensor A: Normal distribution + Sine wave drift
        t2 = progress.add_task("[magenta]Stressing Sensor A (Temperature)...", total=100)
        temp_base = Vector.randn(n_rows).mul_scalar(2.0).add_scalar(25.0) # 25C mean, 2C std
        drift = times.mul_scalar(0.01).sin().mul_scalar(5.0)
        temp = temp_base.add_vec(drift)
        progress.update(t2, advance=100)

        # Sensor B: Correlated pressure
        t3 = progress.add_task("[yellow]Pressurizing Sensor B...", total=100)
        press = temp.mul_scalar(0.1).add_vec(Vector.randn(n_rows).mul_scalar(0.5)).add_scalar(10.0)
        progress.update(t3, advance=100)

        t4 = progress.add_task("[green]Assembling Polars DataFrame & Writing...", total=100)
        df = pl.DataFrame({
            "timestamp": times.to_list(),
            "temperature": temp.to_list(),
            "pressure": press.to_list()
        })
        df.write_csv(out_file)
        progress.update(t4, advance=100)

    console.print(f"[success]Dataset generated successfully: [bold cyan]{out_file}[/bold cyan][/success]")
    console.print(f"[dim]File Size: {n_rows * 24 / (1024*1024):.2f} MB[/dim]")
