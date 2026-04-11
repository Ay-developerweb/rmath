from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.theme import Theme
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import time

# Custom Premium Theme
rmath_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "highlight": "bold magenta",
    "value": "bold white",
    "header": "bold blue underline"
})

console = Console(theme=rmath_theme)

def welcome_banner():
    panel = Panel(
        "[bold cyan]RMATH ANALYZER v0.1.0[/bold cyan]\n"
        "[dim]High-Performance Numerical Toolkit for Data Science & Engineering[/dim]",
        border_style="blue",
        title="[bold white]Welcome[/bold white]",
        subtitle="Powered by Rust & Rayon"
    )
    console.print(panel)

def print_stats_table(data_dict, title="Statistical Summary"):
    table = Table(title=title, border_style="dim")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="value", justify="right")
    
    for k, v in data_dict.items():
        val_str = f"{v:.6f}" if isinstance(v, float) else str(v)
        table.add_row(k, val_str)
    
    console.print(table)

def progress_bar():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    )
