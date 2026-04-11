import argparse
import sys
import os
from .utils.formatter import welcome_banner, console
from .commands.stats import run_stats_analysis
from .commands.engineering import run_spectral_dashboard, run_integration_demo, run_matrix_demo
from .commands.generator import run_data_generation
from .commands.analysis import run_data_analysis
from .commands.geometry import run_spatial_analysis

def main():
    parser = argparse.ArgumentParser(
        description="Rmath Analyzer: Professional Numerical CLI",
        add_help=True
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Operational Modules")

    # --- Generator ---
    gen_p = subparsers.add_parser("generate", help="Synthetic data generation")
    gen_p.add_argument("--out", default="data.csv", help="Output filename")
    gen_p.add_argument("--n", type=int, default=1000000, help="Number of rows")

    # --- Analysis (Polars) ---
    ana_p = subparsers.add_parser("analyze", help="External data analysis (Polars)")
    ana_p.add_argument("file", help="Path to CSV file")
    ana_p.add_argument("--col", required=True, help="Column to analyze")
    ana_p.add_argument("--vs", help="Optional comparison column for T-Test")

    # --- Stats ---
    stats_p = subparsers.add_parser("stats", help="Statistical simulation")
    stats_p.add_argument("--source", choices=["random", "linear"], default="random")
    stats_p.add_argument("--n", type=int, default=100000)

    # --- Engineering ---
    eng_p = subparsers.add_parser("eng", help="Engineering suite")
    eng_p.add_argument("--mode", choices=["fft", "integral", "matrix"], required=True)
    eng_p.add_argument("--n", type=int, default=8192)

    # --- Geometry ---
    geo_p = subparsers.add_parser("geo", help="Spatial analysis")
    geo_p.add_argument("--n", type=int, default=1000, help="Number of points for hull")

    if len(sys.argv) == 1:
        welcome_banner()
        console.print("[dim]Use --help to see available commands[/dim]\n")
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    try:
        if args.command == "generate":
            run_data_generation(args.out, args.n)
        elif args.command == "analyze":
            run_data_analysis(args.file, args.col, args.vs)
        elif args.command == "stats":
            run_stats_analysis(args.source, args.n)
        elif args.command == "eng":
            if args.mode == "fft":
                run_spectral_dashboard(args.n)
            elif args.mode == "integral":
                run_integration_demo()
            elif args.mode == "matrix":
                run_matrix_demo()
        elif args.command == "geo":
            run_spatial_analysis(args.n)
    except Exception as e:
        console.print(f"[error]Execution Error: {str(e)}[/error]")

if __name__ == "__main__":
    main()
