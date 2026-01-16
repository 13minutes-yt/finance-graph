# graph

Professional stock charting application with technical analysis, inspired by Yahoo Finance.

Built with Axum (Rust), DuckDB, and KLineChart.

## Features

- 📊 **Multiple Chart Types**: Candlestick, Line, Area, OHLC
- 📈 **Technical Indicators**: SMA, EMA, Bollinger Bands, RSI, MACD, Stochastic
- ✏️ **Drawing Tools**: Trend lines, Fibonacci retracement, price channels, annotations
- 🌙 **Dark/Light Theme**: Persistent theme preference
- 📁 **CSV Upload**: Import your own OHLCV data
- 💾 **Drawing Persistence**: Save and restore your chart annotations
- 📱 **Responsive**: Works on desktop and mobile

## Run

```bash
cargo run
```

Open <http://localhost:8000>.

## Data

The app loads `data/stocks_extended.csv` (500 days of synthetic data) into `data/data.duckdb` on first run.
Place `klinecharts.min.js` in `static/vendor/` to avoid the CDN dependency.

## API Endpoints

### Candles

```
GET /api/candles?start=YYYY-MM-DD&end=YYYY-MM-DD&limit=N
```

### Indicators

```
GET /api/indicators?start=YYYY-MM-DD&end=YYYY-MM-DD
GET /api/indicators/macd?start=...&end=...&fast=12&slow=26&signal=9
GET /api/indicators/bollinger?start=...&end=...&period=20&std=2
GET /api/indicators/stochastic?start=...&end=...&k=14&d=3
```

### Fibonacci Levels

```
GET /api/fib?start=YYYY-MM-DD HH:MM:SS&end=YYYY-MM-DD HH:MM:SS
```

### Drawings

```
GET /api/drawings
POST /api/drawings (body: {"data": [...]})
```

### Upload CSV

```
POST /api/upload (multipart/form-data, field: "file")
```

CSV must have headers: `timestamp,open,high,low,close,volume`

## Tech Stack

- **Backend**: Rust (Axum framework)
- **Database**: DuckDB (embedded analytical database)
- **Frontend**: Vanilla JavaScript + KLineChart v9.7.0
- **Styling**: CSS with dark/light theme support
