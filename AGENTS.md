# Repository Guidelines

## Project Structure & Module Organization
- `src/main.rs` contains the Axum server, DuckDB setup, and API handlers.
- `static/` holds the frontend (`index.html`, `app.js`, `styles.css`) served at `/`.
- `data/` stores CSV inputs (`stocks_extended.csv` preferred, `stocks.csv` fallback) and `data.duckdb` (created on first run).
- `README.md` provides the quickstart and current API endpoints.

## Build, Test, and Development Commands
- `cargo run` starts the server at `http://localhost:8000` and serves the UI.
- `cargo build` builds the binary for local verification.
- `cargo test` runs Rust tests (none are defined yet).
- `cargo fmt` formats Rust code if `rustfmt` is installed.

## Coding Style & Naming Conventions
- Rust: follow `rustfmt` defaults; use `snake_case` for functions/vars and `PascalCase` for structs/enums.
- Frontend: keep 2-space indentation in `static/` files; use `camelCase` in JS and `kebab-case` for CSS classes (e.g., `.page-header`).
- Keep API routes under `/api` and return JSON shapes that match the frontend expectations.

## Testing Guidelines
- Automated tests are not present; validate changes manually.
- Suggested checks:
  - Run `cargo run` and load the UI.
  - Verify `/api/candles`, `/api/indicators`, and `/api/fib` with `curl` to confirm data flow.

## Data & Configuration Notes
- On first run, `data/stocks_extended.csv` is imported into `data/data.duckdb` when present (otherwise `data/stocks.csv`).
- To re-import fresh CSV data, delete `data/data.duckdb` and restart the server.
- Avoid committing large generated `data.duckdb` files if adding version control.

## Commit & Pull Request Guidelines
- No git history is available in this workspace, so there are no established commit conventions.
- If you initialize git, use concise, imperative commit subjects (e.g., `Add fib endpoint`).
- PRs should include a short description, testing notes, and screenshots for UI changes.
