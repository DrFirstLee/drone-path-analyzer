# drone-path-analyzer

Analyze a drone flight path CSV and export segmented CSV files plus an `AI Analyze Overview.png` image.

## Install

```bash
pip install drone-path-analyzer
```

## Usage

```bash
drone-path-analyzer flight.csv
```

By default, results are written to `output/`:

```text
output/
  AI Analyze Overview.png
  full_result.csv
  splited/
    line-1.csv
    curve-2.csv
    circle-3.csv
```

You can choose a different output folder or sliding window size:

```bash
drone-path-analyzer flight.csv --output results --sliding-window 30
```

The input CSV can include extra leading columns. The analyzer looks for a timestamp-like column and reads `Time`, `Longitude`, `Latitude`, and optionally `Altitude` from that point.
