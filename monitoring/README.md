# NIDS Monitoring Stack

Prometheus + Grafana monitoring for the Network Intrusion Detection System.

## Quick Start

```bash
# From the project root
docker-compose up -d
```

This starts three services:

| Service    | URL                     | Purpose                        |
|------------|-------------------------|--------------------------------|
| NIDS API   | http://localhost:8000   | ML prediction service          |
| Prometheus | http://localhost:9090   | Metrics scraping & storage     |
| Grafana    | http://localhost:3000   | Dashboards & visualization     |

## Grafana Login

- **Username:** `admin`
- **Password:** `nids-admin`

The NIDS dashboard loads automatically as the home dashboard.

## Verify Metrics Are Flowing

1. Open Prometheus targets: http://localhost:9090/targets
2. Confirm `nids-api` shows state = **UP**
3. Open Grafana: http://localhost:3000
4. The NIDS dashboard should show live data within 15 seconds

## Dashboard Panels

| Panel                      | Metric                                | Type       |
|----------------------------|---------------------------------------|------------|
| Total Predictions          | `nids_predictions_total`              | Stat       |
| Attacks Detected           | `nids_attacks_detected_total`         | Stat       |
| Attack Rate %              | attacks / predictions * 100           | Stat       |
| Avg Prediction Latency     | `nids_prediction_latency_seconds`     | Stat       |
| Predictions Over Time      | rate by `prediction_class`            | Time series|
| Attacks Over Time          | rate by `attack_type`                 | Bar chart  |
| Prediction Latency p50/95/99 | histogram quantiles                 | Time series|
| API Request Rate           | `http_requests_total`                 | Time series|
| Attack Type Distribution   | by `attack_type`                      | Pie chart  |
| Model Status               | `nids_models_loaded`                  | Stat       |
| API Uptime                 | `process_uptime_seconds`              | Stat       |
| Top Attack Types           | top 10 by count                       | Table      |

## File Structure

```
monitoring/
  prometheus/
    prometheus.yml          # Scrape config (targets nids-api:8000)
  grafana/
    provisioning/
      datasources/
        prometheus.yml      # Auto-register Prometheus datasource
      dashboards/
        dashboard.yml       # Dashboard file provider config
    dashboards/
      nids-dashboard.json   # Pre-built NIDS dashboard
```

## Useful Commands

```bash
# View logs
docker-compose logs -f grafana
docker-compose logs -f prometheus

# Restart a single service
docker-compose restart grafana

# Reset Grafana (clear all customizations)
docker-compose down
docker volume rm fyp-nids-ml_grafana-data
docker-compose up -d

# Check Prometheus config is valid
docker-compose exec prometheus promtool check config /etc/prometheus/prometheus.yml
```

## Data Retention

- Prometheus retains metrics for **30 days** (configurable via `--storage.tsdb.retention.time`)
- Grafana data (dashboards, preferences) persists in the `grafana-data` Docker volume

## Running Without Docker

If running the API locally (not in Docker), update `monitoring/prometheus/prometheus.yml`:

```yaml
static_configs:
  - targets: ["host.docker.internal:8000"]   # Instead of "nids-api:8000"
```

Then start only Prometheus and Grafana:

```bash
docker-compose up -d prometheus grafana
```
