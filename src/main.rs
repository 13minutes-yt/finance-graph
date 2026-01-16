use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use axum::extract::{DefaultBodyLimit, Multipart, Query, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use duckdb::{params, Connection};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tower_http::services::ServeDir;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone)]
struct AppState {
    db: Arc<Mutex<Connection>>,
}

#[derive(Serialize, Clone)]
struct Candle {
    timestamp: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

#[derive(Serialize)]
struct IndicatorPoint {
    timestamp: String,
    sma_14: Option<f64>,
    ema_14: Option<f64>,
    rsi_14: Option<f64>,
}

#[derive(Serialize)]
struct MacdPoint {
    timestamp: String,
    macd: Option<f64>,
    signal: Option<f64>,
    histogram: Option<f64>,
}

#[derive(Serialize)]
struct BollingerPoint {
    timestamp: String,
    middle: Option<f64>,
    upper: Option<f64>,
    lower: Option<f64>,
}

#[derive(Serialize)]
struct StochasticPoint {
    timestamp: String,
    k: Option<f64>,
    d: Option<f64>,
}

#[derive(Serialize)]
struct FibLevels {
    low: f64,
    high: f64,
    levels: Vec<FibLevel>,
}

#[derive(Serialize)]
struct FibLevel {
    ratio: f64,
    value: f64,
}

#[derive(Serialize)]
struct UploadResponse {
    rows: i64,
}

#[derive(Serialize)]
struct DrawingState {
    data: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct CandleQuery {
    start: Option<String>,
    end: Option<String>,
    limit: Option<u32>,
}

#[derive(Deserialize)]
struct RangeQuery {
    start: Option<String>,
    end: Option<String>,
}

#[derive(Deserialize)]
struct MacdQuery {
    start: Option<String>,
    end: Option<String>,
    fast: Option<u32>,
    slow: Option<u32>,
    signal: Option<u32>,
}

#[derive(Deserialize)]
struct BollingerQuery {
    start: Option<String>,
    end: Option<String>,
    period: Option<u32>,
    std: Option<f64>,
}

#[derive(Deserialize)]
struct StochasticQuery {
    start: Option<String>,
    end: Option<String>,
    k: Option<u32>,
    d: Option<u32>,
}

#[derive(Deserialize)]
struct DrawingPayload {
    data: serde_json::Value,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "graph=debug,tower_http=debug".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let db_path = Path::new("data/data.duckdb");
    let csv_path = if Path::new("data/stocks_extended.csv").exists() {
        Path::new("data/stocks_extended.csv")
    } else {
        Path::new("data/stocks.csv")
    };
    let conn = Connection::open(db_path).context("open DuckDB")?;
    initialize_db(&conn, csv_path).context("init DuckDB")?;

    let state = AppState {
        db: Arc::new(Mutex::new(conn)),
    };

    let app = Router::new()
        .route("/api/candles", get(get_candles))
        .route("/api/indicators", get(get_indicators))
        .route("/api/indicators/macd", get(get_macd))
        .route("/api/indicators/bollinger", get(get_bollinger))
        .route("/api/indicators/stochastic", get(get_stochastic))
        .route("/api/drawings", get(get_drawings).post(save_drawings))
        .route("/api/upload", post(upload_csv))
        .route("/api/fib", get(get_fib))
        .nest_service("/", ServeDir::new("static"))
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024))
        .with_state(state);

    let addr: SocketAddr = "0.0.0.0:8000".parse()?;
    tracing::info!("listening on {addr}");
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
    Ok(())
}

fn initialize_db(conn: &Connection, csv_path: &Path) -> anyhow::Result<()> {
    let interval_exists: i64 = conn.query_row(
        "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'candles' AND column_name = 'interval'",
        [],
        |row| row.get(0),
    )?;

    if interval_exists == 0 {
        conn.execute_batch("DROP TABLE IF EXISTS candles;")?;
    }

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS candles (
            timestamp TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            \"interval\" TEXT DEFAULT 'day'
        );",
    )?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS drawings (
            id INTEGER PRIMARY KEY,
            data TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );",
    )?;

    let existing: i64 = conn.query_row("SELECT COUNT(*) FROM candles", [], |row| row.get(0))?;
    if existing == 0 {
        let csv_str = csv_path
            .to_str()
            .context("CSV path not valid UTF-8")?
            .replace('\\', "/");
        let sql = format!(
            "COPY candles (timestamp, open, high, low, close, volume) FROM '{}' (HEADER, AUTO_DETECT TRUE);",
            csv_str
        );
        conn.execute_batch(&sql)?;
    }
    Ok(())
}

async fn get_candles(
    State(state): State<AppState>,
    Query(query): Query<CandleQuery>,
) -> Result<Json<Vec<Candle>>, (StatusCode, String)> {
    let (start, end) = normalize_query_bounds(query.start, query.end);
    let limit = query.limit.unwrap_or(500) as i64;
    let conn = state.db.lock().await;
    let mut stmt = conn
        .prepare(
            "SELECT
                strftime(timestamp, '%Y-%m-%d %H:%M:%S') AS ts,
                open, high, low, close, volume
             FROM candles
             WHERE (?1 IS NULL OR timestamp >= ?1)
               AND (?2 IS NULL OR timestamp <= ?2)
             ORDER BY timestamp
             LIMIT ?3",
        )
        .map_err(internal_error)?;
    let mut rows = stmt
        .query(params![start.as_deref(), end.as_deref(), limit])
        .map_err(internal_error)?;
    let mut candles = Vec::new();
    while let Some(row) = rows.next().map_err(internal_error)? {
        candles.push(Candle {
            timestamp: row.get(0).map_err(internal_error)?,
            open: row.get(1).map_err(internal_error)?,
            high: row.get(2).map_err(internal_error)?,
            low: row.get(3).map_err(internal_error)?,
            close: row.get(4).map_err(internal_error)?,
            volume: row.get(5).map_err(internal_error)?,
        });
    }
    Ok(Json(candles))
}

async fn get_indicators(
    State(state): State<AppState>,
    Query(query): Query<RangeQuery>,
) -> Result<Json<Vec<IndicatorPoint>>, (StatusCode, String)> {
    let (start, end) = normalize_query_bounds(query.start, query.end);
    let conn = state.db.lock().await;
    let candles = load_candles_for_indicators(&conn, end.as_deref()).map_err(internal_error)?;
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let sma_14 = sma(&closes, 14);
    let ema_14 = ema(&closes, 14);
    let rsi_14 = rsi(&closes, 14);

    let mut points = Vec::with_capacity(candles.len());
    for (idx, candle) in candles.iter().enumerate() {
        points.push(IndicatorPoint {
            timestamp: candle.timestamp.clone(),
            sma_14: sma_14[idx],
            ema_14: ema_14[idx],
            rsi_14: rsi_14[idx],
        });
    }

    let filtered = filter_by_start(points, start.as_deref());
    Ok(Json(filtered))
}

async fn get_macd(
    State(state): State<AppState>,
    Query(query): Query<MacdQuery>,
) -> Result<Json<Vec<MacdPoint>>, (StatusCode, String)> {
    let fast = query.fast.unwrap_or(12) as usize;
    let slow = query.slow.unwrap_or(26) as usize;
    let signal = query.signal.unwrap_or(9) as usize;
    if fast == 0 || slow == 0 || signal == 0 || fast >= slow {
        return Err(bad_request("Invalid MACD parameters"));
    }

    let (start, end) = normalize_query_bounds(query.start, query.end);
    let conn = state.db.lock().await;
    let candles = load_candles_for_indicators(&conn, end.as_deref()).map_err(internal_error)?;
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let (macd_line, signal_line, histogram) = macd(&closes, fast, slow, signal);

    let mut points = Vec::with_capacity(candles.len());
    for (idx, candle) in candles.iter().enumerate() {
        points.push(MacdPoint {
            timestamp: candle.timestamp.clone(),
            macd: macd_line[idx],
            signal: signal_line[idx],
            histogram: histogram[idx],
        });
    }

    let filtered = filter_by_start(points, start.as_deref());
    Ok(Json(filtered))
}

async fn get_bollinger(
    State(state): State<AppState>,
    Query(query): Query<BollingerQuery>,
) -> Result<Json<Vec<BollingerPoint>>, (StatusCode, String)> {
    let period = query.period.unwrap_or(20) as usize;
    let std = query.std.unwrap_or(2.0);
    if period == 0 || std <= 0.0 {
        return Err(bad_request("Invalid Bollinger parameters"));
    }

    let (start, end) = normalize_query_bounds(query.start, query.end);
    let conn = state.db.lock().await;
    let candles = load_candles_for_indicators(&conn, end.as_deref()).map_err(internal_error)?;
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let (middle, upper, lower) = bollinger(&closes, period, std);

    let mut points = Vec::with_capacity(candles.len());
    for (idx, candle) in candles.iter().enumerate() {
        points.push(BollingerPoint {
            timestamp: candle.timestamp.clone(),
            middle: middle[idx],
            upper: upper[idx],
            lower: lower[idx],
        });
    }

    let filtered = filter_by_start(points, start.as_deref());
    Ok(Json(filtered))
}

async fn get_stochastic(
    State(state): State<AppState>,
    Query(query): Query<StochasticQuery>,
) -> Result<Json<Vec<StochasticPoint>>, (StatusCode, String)> {
    let k_period = query.k.unwrap_or(14) as usize;
    let d_period = query.d.unwrap_or(3) as usize;
    if k_period == 0 || d_period == 0 {
        return Err(bad_request("Invalid stochastic parameters"));
    }

    let (start, end) = normalize_query_bounds(query.start, query.end);
    let conn = state.db.lock().await;
    let candles = load_candles_for_indicators(&conn, end.as_deref()).map_err(internal_error)?;
    let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
    let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let (k_values, d_values) = stochastic(&highs, &lows, &closes, k_period, d_period);

    let mut points = Vec::with_capacity(candles.len());
    for (idx, candle) in candles.iter().enumerate() {
        points.push(StochasticPoint {
            timestamp: candle.timestamp.clone(),
            k: k_values[idx],
            d: d_values[idx],
        });
    }

    let filtered = filter_by_start(points, start.as_deref());
    Ok(Json(filtered))
}

async fn get_drawings(
    State(state): State<AppState>,
) -> Result<Json<DrawingState>, (StatusCode, String)> {
    let conn = state.db.lock().await;
    let mut stmt = conn
        .prepare("SELECT data FROM drawings WHERE id = 1")
        .map_err(internal_error)?;
    let mut rows = stmt.query([]).map_err(internal_error)?;
    if let Some(row) = rows.next().map_err(internal_error)? {
        let data: String = row.get(0).map_err(internal_error)?;
        let parsed = serde_json::from_str(&data).map_err(internal_error)?;
        Ok(Json(DrawingState {
            data: Some(parsed),
        }))
    } else {
        Ok(Json(DrawingState { data: None }))
    }
}

async fn save_drawings(
    State(state): State<AppState>,
    Json(payload): Json<DrawingPayload>,
) -> Result<StatusCode, (StatusCode, String)> {
    let data = payload.data.to_string();
    let conn = state.db.lock().await;
    conn.execute("DELETE FROM drawings", [])
        .map_err(internal_error)?;
    conn.execute(
        "INSERT INTO drawings (id, data, updated_at) VALUES (1, ?, CURRENT_TIMESTAMP)",
        params![data],
    )
    .map_err(internal_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn upload_csv(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<UploadResponse>, (StatusCode, String)> {
    let mut file_bytes = None;
    while let Some(field) = multipart.next_field().await.map_err(bad_request)? {
        if field.name() == Some("file") {
            let bytes = field.bytes().await.map_err(bad_request)?;
            file_bytes = Some(bytes);
            break;
        }
    }

    let bytes = file_bytes.ok_or_else(|| bad_request("Missing file field"))?;
    let header = bytes
        .split(|b| *b == b'\n')
        .next()
        .unwrap_or_default();
    let header_string = String::from_utf8_lossy(header).to_string();
    let header_str = header_string.trim().trim_end_matches('\r');
    if header_str != "timestamp,open,high,low,close,volume" {
        return Err(bad_request(
            "Invalid CSV header. Expected: timestamp,open,high,low,close,volume",
        ));
    }

    let upload_path = Path::new("data/upload.csv");
    tokio::fs::write(upload_path, &bytes)
        .await
        .map_err(internal_error)?;

    let conn = state.db.lock().await;
    conn.execute("DELETE FROM candles", [])
        .map_err(internal_error)?;
    let csv_str = upload_path
        .to_str()
        .context("upload path not valid UTF-8")
        .map_err(internal_error)?
        .replace('\\', "/");
    let sql = format!(
        "COPY candles (timestamp, open, high, low, close, volume) FROM '{}' (HEADER, AUTO_DETECT TRUE);",
        csv_str
    );
    conn.execute_batch(&sql).map_err(internal_error)?;
    let rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM candles", [], |row| row.get(0))
        .map_err(internal_error)?;

    let _ = tokio::fs::remove_file(upload_path).await;

    Ok(Json(UploadResponse { rows }))
}

async fn get_fib(
    State(state): State<AppState>,
    Query(query): Query<RangeQuery>,
) -> Result<Json<FibLevels>, (StatusCode, String)> {
    let (start, end) = normalize_query_bounds(query.start, query.end);
    let conn = state.db.lock().await;
    let (low, high): (f64, f64) = match (start.as_deref(), end.as_deref()) {
        (Some(start), Some(end)) => conn
            .query_row(
                "SELECT min(low), max(high) FROM candles WHERE timestamp BETWEEN ? AND ?",
                params![start, end],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(internal_error)?,
        _ => conn
            .query_row(
                "SELECT min(low), max(high) FROM candles",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(internal_error)?,
    };

    let levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        .into_iter()
        .map(|ratio| FibLevel {
            ratio,
            value: high - (high - low) * ratio,
        })
        .collect();

    Ok(Json(FibLevels { low, high, levels }))
}

fn normalize_query_bounds(
    start: Option<String>,
    end: Option<String>,
) -> (Option<String>, Option<String>) {
    let start = start.map(|value| normalize_bound(&value, false));
    let end = end.map(|value| normalize_bound(&value, true));
    (start, end)
}

fn normalize_bound(value: &str, is_end: bool) -> String {
    if value.len() == 10 {
        if is_end {
            format!("{} 23:59:59", value)
        } else {
            format!("{} 00:00:00", value)
        }
    } else {
        value.to_string()
    }
}

fn load_candles_for_indicators(
    conn: &Connection,
    end: Option<&str>,
) -> anyhow::Result<Vec<Candle>> {
    let mut stmt = conn.prepare(
        "SELECT
            strftime(timestamp, '%Y-%m-%d %H:%M:%S') AS ts,
            open, high, low, close, volume
         FROM candles
         WHERE (?1 IS NULL OR timestamp <= ?1)
         ORDER BY timestamp",
    )?;
    let mut rows = stmt.query(params![end])?;
    let mut candles = Vec::new();
    while let Some(row) = rows.next()? {
        candles.push(Candle {
            timestamp: row.get(0)?,
            open: row.get(1)?,
            high: row.get(2)?,
            low: row.get(3)?,
            close: row.get(4)?,
            volume: row.get(5)?,
        });
    }
    Ok(candles)
}

fn filter_by_start<T>(mut points: Vec<T>, start: Option<&str>) -> Vec<T>
where
    T: HasTimestamp,
{
    if let Some(start) = start {
        points.retain(|point| point.timestamp() >= start);
    }
    points
}

trait HasTimestamp {
    fn timestamp(&self) -> &str;
}

impl HasTimestamp for IndicatorPoint {
    fn timestamp(&self) -> &str {
        &self.timestamp
    }
}

impl HasTimestamp for MacdPoint {
    fn timestamp(&self) -> &str {
        &self.timestamp
    }
}

impl HasTimestamp for BollingerPoint {
    fn timestamp(&self) -> &str {
        &self.timestamp
    }
}

impl HasTimestamp for StochasticPoint {
    fn timestamp(&self) -> &str {
        &self.timestamp
    }
}

fn sma(values: &[f64], period: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    if period == 0 {
        return output;
    }
    let mut sum = 0.0;
    for i in 0..values.len() {
        sum += values[i];
        if i >= period {
            sum -= values[i - period];
        }
        if i + 1 >= period {
            output[i] = Some(sum / period as f64);
        }
    }
    output
}

fn ema(values: &[f64], period: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    if values.is_empty() || period == 0 {
        return output;
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut prev = values[0];
    output[0] = Some(prev);
    for i in 1..values.len() {
        let next = values[i] * alpha + prev * (1.0 - alpha);
        output[i] = Some(next);
        prev = next;
    }
    output
}

fn rsi(values: &[f64], period: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    if values.len() < 2 || period == 0 {
        return output;
    }
    let mut gains = vec![0.0; values.len()];
    let mut losses = vec![0.0; values.len()];
    for i in 1..values.len() {
        let delta = values[i] - values[i - 1];
        if delta > 0.0 {
            gains[i] = delta;
        } else {
            losses[i] = -delta;
        }
    }

    let mut gain_sum = 0.0;
    let mut loss_sum = 0.0;
    for i in 1..values.len() {
        gain_sum += gains[i];
        loss_sum += losses[i];
        if i >= period {
            gain_sum -= gains[i - period];
            loss_sum -= losses[i - period];
        }
        if i >= period {
            let avg_gain = gain_sum / period as f64;
            let avg_loss = loss_sum / period as f64;
            if avg_loss == 0.0 {
                output[i] = None;
            } else {
                let rs = avg_gain / avg_loss;
                output[i] = Some(100.0 - (100.0 / (1.0 + rs)));
            }
        }
    }
    output
}

fn macd(
    values: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
) -> (Vec<Option<f64>>, Vec<Option<f64>>, Vec<Option<f64>>) {
    let len = values.len();
    let mut macd_line = vec![None; len];
    let mut signal_line = vec![None; len];
    let mut histogram = vec![None; len];
    if len == 0 || fast == 0 || slow == 0 || signal == 0 {
        return (macd_line, signal_line, histogram);
    }

    let fast_ema = ema(values, fast);
    let slow_ema = ema(values, slow);
    for i in 0..len {
        if i + 1 >= slow {
            if let (Some(fast_val), Some(slow_val)) = (fast_ema[i], slow_ema[i]) {
                macd_line[i] = Some(fast_val - slow_val);
            }
        }
    }

    let alpha = 2.0 / (signal as f64 + 1.0);
    let mut prev: Option<f64> = None;
    let mut count = 0usize;
    for i in 0..len {
        if let Some(macd_val) = macd_line[i] {
            count += 1;
            let next = match prev {
                None => macd_val,
                Some(prev_val) => macd_val * alpha + prev_val * (1.0 - alpha),
            };
            prev = Some(next);
            if count >= signal {
                signal_line[i] = Some(next);
            }
        }

        if let (Some(macd_val), Some(signal_val)) = (macd_line[i], signal_line[i]) {
            histogram[i] = Some(macd_val - signal_val);
        }
    }

    (macd_line, signal_line, histogram)
}

fn bollinger(
    values: &[f64],
    period: usize,
    std_multiplier: f64,
) -> (Vec<Option<f64>>, Vec<Option<f64>>, Vec<Option<f64>>) {
    let len = values.len();
    let mut middle = vec![None; len];
    let mut upper = vec![None; len];
    let mut lower = vec![None; len];
    if len == 0 || period == 0 {
        return (middle, upper, lower);
    }

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for i in 0..len {
        sum += values[i];
        sum_sq += values[i] * values[i];
        if i >= period {
            sum -= values[i - period];
            sum_sq -= values[i - period] * values[i - period];
        }
        if i + 1 >= period {
            let mean = sum / period as f64;
            let variance = (sum_sq / period as f64) - (mean * mean);
            let std = variance.max(0.0).sqrt();
            middle[i] = Some(mean);
            upper[i] = Some(mean + std_multiplier * std);
            lower[i] = Some(mean - std_multiplier * std);
        }
    }

    (middle, upper, lower)
}

fn stochastic(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    k_period: usize,
    d_period: usize,
) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    let len = closes.len();
    let mut k_values = vec![None; len];
    let mut d_values = vec![None; len];
    if len == 0 || k_period == 0 || d_period == 0 {
        return (k_values, d_values);
    }

    for i in 0..len {
        if i + 1 >= k_period {
            let start = i + 1 - k_period;
            let mut highest = highs[start];
            let mut lowest = lows[start];
            for j in start..=i {
                if highs[j] > highest {
                    highest = highs[j];
                }
                if lows[j] < lowest {
                    lowest = lows[j];
                }
            }
            if (highest - lowest).abs() > f64::EPSILON {
                let value = (closes[i] - lowest) / (highest - lowest) * 100.0;
                k_values[i] = Some(value);
            }
        }
    }

    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..len {
        if let Some(value) = k_values[i] {
            sum += value;
            count += 1;
        }
        if i >= d_period {
            if let Some(old) = k_values[i - d_period] {
                sum -= old;
                count -= 1;
            }
        }
        if i + 1 >= d_period && count == d_period {
            d_values[i] = Some(sum / d_period as f64);
        }
    }

    (k_values, d_values)
}

fn bad_request(error: impl std::fmt::Display) -> (StatusCode, String) {
    (StatusCode::BAD_REQUEST, error.to_string())
}

fn internal_error(error: impl std::fmt::Display) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
}
