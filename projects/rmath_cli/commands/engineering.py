from rmath.signal import fft_styled, convolve
from rmath.vector import Vector
from rmath.calculus import integrate_simpson
from rmath.array import Array
from ..utils.formatter import console, progress_bar, Panel
import math

def run_spectral_dashboard(n_samples=8192):
    with progress_bar() as progress:
        t1 = progress.add_task("[cyan]Synthesizing Signal...", total=100)
        t = Vector.linspace(0, 1.0, n_samples)
        s1 = t.mul_scalar(2.0 * math.pi * 50.0).sin()
        s2 = t.mul_scalar(2.0 * math.pi * 120.0).sin().mul_scalar(0.5)
        signal_v = s1.add_vec(s2)
        progress.update(t1, advance=100)
        
        t2 = progress.add_task("[magenta]Executing FFT...", total=100)
        mags, phases = fft_styled(signal_v)
        progress.update(t2, advance=100)

    half = mags.head(n_samples // 2)
    peak_idx = half.argmax()
    peak_freq = peak_idx
    
    console.print(Panel(
        f"Primary Frequency: [bold green]{peak_freq} Hz[/bold green]\n"
        f"Spectral Power at Peak: {mags.to_list()[peak_idx]:.2f}",
        title="Spectral Analysis Results",
        border_style="magenta"
    ))

def run_integration_demo(lower=0.0, upper=1.0):
    console.print(f"[info]Integrating f(x) = sin(x) from {lower} to {upper}...[/info]")
    result = integrate_simpson(math.sin, lower, upper, 1_000_000)
    expected = -math.cos(upper) - (-math.cos(lower))
    error = abs(result - expected)
    
    console.print(f"Result: [bold white]{result:.10f}[/bold white]")
    console.print(f"Absolute Error: [dim]{error:.2e}[/dim]")

def run_matrix_demo():
    console.print("[info]Initializing 3x3 System of Equations...[/info]")
    # A * x = b
    # 2x + y - z = 8
    # -3x - y + 2z = -11
    # -2x + y + 2z = -3
    
    mat_data = [
        [2.0, 1.0, -1.0],
        [-3.0, -1.0, 2.0],
        [-2.0, 1.0, 2.0]
    ]
    b_data = [8.0, -11.0, -3.0]
    
    a = Array(mat_data)
    # Convert Vector to Array for the solver
    b = Array([[x] for x in b_data])
    
    with progress_bar() as progress:
        t1 = progress.add_task("[magenta]Solving Linear System...", total=100)
        # solve returns an Array (column vector)
        x_mat = a.solve(b)
        progress.update(t1, advance=100)
    
    sol = x_mat.to_list() # This might be [[x], [y], [z]]
    console.print(Panel(
        f"x = [bold cyan]{sol[0][0]:.2f}[/bold cyan]\n"
        f"y = [bold cyan]{sol[1][0]:.2f}[/bold cyan]\n"
        f"z = [bold cyan]{sol[2][0]:.2f}[/bold cyan]",
        title="Solution Found (Gaussian Elimination)",
        border_style="cyan"
    ))
