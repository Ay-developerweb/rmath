from rmath.geometry import convex_hull, is_point_in_polygon
from ..utils.formatter import console, progress_bar, Panel
from rich.table import Table
import random

def run_spatial_analysis(n_points=500):
    console.print(f"[info]Generating {n_points} random spatial coordinates...[/info]")
    
    px = [random.uniform(0, 100) for _ in range(n_points)]
    py = [random.uniform(0, 100) for _ in range(n_points)]
    
    with progress_bar() as progress:
        t1 = progress.add_task("[cyan]Computing Convex Hull...", total=100)
        # convex_hull returns (hx, hy) directly
        hx, hy = convex_hull(px, py)
        progress.update(t1, advance=100)
        
    table = Table(title="Convex Hull Boundary Points", border_style="magenta")
    table.add_column("Vertex", style="dim")
    table.add_column("X-Coord", justify="right")
    table.add_column("Y-Coord", justify="right")
    
    for i in range(min(10, len(hx))):
        table.add_row(str(i), f"{hx[i]:.2f}", f"{hy[i]:.2f}")
    
    if len(hx) > 10:
        table.add_row("...", "...", "...")
        
    console.print(table)
    console.print(f"[success]Hull complete. {len(hx)} vertices found.[/success]")
    
    # Selection check
    test_x, test_y = 50.0, 50.0
    is_inside = is_point_in_polygon(test_x, test_y, hx, hy)
    
    console.print(Panel(
        f"Test Point: ({test_x}, {test_y})\n"
        f"Result: {'[success]INSIDE HULL[/success]' if is_inside else '[error]OUTSIDE HULL[/error]'}",
        title="Spatial Containment Check",
        border_style="blue"
    ))
