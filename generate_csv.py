"""
Entry point — generates output.csv from the periocular image folder.

Usage:
    python generate_csv.py
"""

from src.dataset import generate_csv

if __name__ == "__main__":
    generate_csv()
