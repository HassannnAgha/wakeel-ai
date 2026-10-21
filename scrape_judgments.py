"""Download Pakistani court judgments into pdfs/.

Currently supports the Lahore High Court, which serves judgments at a predictable
URL:  https://sys.lhc.gov.pk/appjudgments/<YEAR>LHC<N>.pdf

Numbering is sparse - most N return a 404 HTML page rather than a PDF - so the
scraper sweeps a range and keeps only real PDFs.

    python scrape_judgments.py --year 2023 --start 100 --end 700
    python scrape_judgments.py --year 2023 --start 100 --end 200 --out pdfs

These are public court records. Be polite: keep the worker count low so the
court's server is not hammered.
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

LHC_URL = "https://sys.lhc.gov.pk/appjudgments/{year}LHC{number}.pdf"
TIMEOUT = 30


def fetch_one(year: int, number: int, out_dir: Path) -> tuple:
    """Return (number, path or None, reason)."""
    url = LHC_URL.format(year=year, number=number)
    target = out_dir / f"lhc{year}_{number}.pdf"
    if target.exists() and target.stat().st_size > 0:
        return number, target, "cached"
    try:
        resp = requests.get(url, timeout=TIMEOUT)
    except Exception as e:
        return number, None, type(e).__name__
    if resp.status_code != 200:
        return number, None, f"http {resp.status_code}"
    # 404s are served as an HTML page with a 200 in some cases, so check content.
    if not resp.content.startswith(b"%PDF"):
        return number, None, "not a pdf"
    target.write_bytes(resp.content)
    return number, target, f"{len(resp.content)} bytes"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--end", type=int, default=300)
    ap.add_argument("--out", default="pdfs", help="output directory (default: pdfs)")
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel downloads (keep this modest)")
    args = ap.parse_args()

    if args.end < args.start:
        raise SystemExit("--end must not be less than --start")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    numbers = range(args.start, args.end + 1)
    print(f"Sweeping {args.year}LHC{args.start}..{args.end} "
          f"({len(numbers)} candidates) into {out_dir}/")

    found = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, args.year, n, out_dir): n for n in numbers}
        for fut in as_completed(futures):
            number, path, reason = fut.result()
            if path:
                found += 1
                print(f"  {args.year}LHC{number}: {reason}")

    total = len(list(out_dir.glob('*.pdf')))
    print(f"\nDownloaded {found} judgment(s); {total} PDF(s) now in {out_dir}/")
    if found:
        print("Next: python wakeel_ai.py   (or ui.py) to index them.")
    else:
        print("Nothing found. Most numbers in a range do not exist - try a wider range.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
