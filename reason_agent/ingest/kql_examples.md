# KQL Examples for Azure Data Explorer (ADX)

This document provides KQL (Kusto Query Language) examples that mirror the local temporal feature extraction, enabling seamless lift-and-shift to Azure production.

**Why KQL/ADX?** Azure Data Explorer provides built-in time-series anomaly detection, forecasting, and root cause analysis (RCA) functions optimized for billions of events.

**Source:** [Azure Data Explorer anomaly detection](https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/anomaly-detection) and [KQL RCA functions](https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/root-cause-analysis)

---

## 1. Rolling Window Aggregations

Local equivalent: `TemporalFeatureExtractor.compute_rolling_windows()`

```kql
// Count events per user in rolling 7/14/30 day windows
SecurityEvents
| where timestamp > ago(30d)
| summarize
    count_7d = countif(timestamp > ago(7d)),
    count_14d = countif(timestamp > ago(14d)),
    count_30d = count()
    by user_id
```

---

## 2. Burstiness Detection

Local equivalent: `TemporalFeatureExtractor.compute_burstiness()`

```kql
// Detect signup storms (burst of signups from same IP)
SecurityEvents
| where event_type == "signup"
| summarize events = count() by ip_address, bin(timestamp, 10m)
| where events > 10  // Threshold: >10 signups in 10 minutes
| project ip_address, signup_burst = events, time_window = timestamp
```

---

## 3. Velocity Rule Violations

Local equivalent: `TemporalFeatureExtractor.compute_velocity_features()`

```kql
// Detect accounts exceeding login velocity limits
SecurityEvents
| where event_type == "login"
| where timestamp > ago(1h)
| summarize login_count = count() by user_id
| where login_count > 50  // Threshold: >50 logins per hour
| project user_id, login_count, severity = "High"
```

---

## 4. Exponential Decay Weighted Count

Local equivalent: `TemporalFeatureExtractor.compute_exponential_decay_count()`

```kql
// Weight recent events more heavily using exponential decay
let lambda = 0.1;
SecurityEvents
| extend days_ago = datetime_diff('day', now(), timestamp)
| extend weight = exp(-lambda * days_ago)
| summarize decay_weighted_count = sum(weight) by user_id
| order by decay_weighted_count desc
```

---

## 5. Time-Series Anomaly Detection (Built-in)

**KQL has native anomaly detection using `series_decompose_anomalies()`**

```kql
// Detect anomalies in login patterns over time
SecurityEvents
| where event_type == "login"
| make-series login_count = count() default=0 on timestamp step 1h by user_id
| extend anomalies = series_decompose_anomalies(login_count, 1.5)  // 1.5 = sensitivity
| mv-expand timestamp to typeof(datetime), login_count to typeof(long), anomalies to typeof(double)
| where anomalies != 0  // Non-zero = anomaly
| project user_id, timestamp, login_count, anomaly_score = anomalies
```

**Why this is powerful:** Native time-series decomposition (trend, seasonality, anomalies) at scale.

---

## 6. Root Cause Analysis (RCA)

**KQL RCA functions identify correlated dimensions for anomalies**

```kql
// Find root cause of login failures spike
SecurityEvents
| where event_type == "login_failure"
| where timestamp > ago(24h)
| evaluate autocluster()  // Automatically find patterns
```

**Output:** Shows which combinations of (ip_address, geo_location, user_agent, etc.) are over-represented in failures.

---

## 7. Shared Device/IP Detection

Local equivalent: `BiTemporalGraphLoader.detect_shared_device_clusters()`

```kql
// Find devices shared by multiple accounts
SecurityEvents
| summarize accounts = dcount(user_id) by device_id
| where accounts >= 5  // Threshold: >=5 accounts per device
| join kind=inner (
    SecurityEvents
    | project user_id, device_id, timestamp
) on device_id
| summarize account_list = make_set(user_id) by device_id
| project device_id, shared_account_count = array_length(account_list), account_list
```

---

## 8. Impossible Travel Detection

```kql
// Detect logins from geographically impossible locations in short time
let MaxKmPerHour = 1000;  // Speed of commercial flight
SecurityEvents
| where event_type == "login"
| extend prev_geo = prev(geo_location, 1), prev_time = prev(timestamp, 1)
| extend time_diff_hours = datetime_diff('hour', timestamp, prev_time)
| extend geo_distance_km = geo_distance_2points(
    geo_longitude(geo_location), geo_latitude(geo_location),
    geo_longitude(prev_geo), geo_latitude(prev_geo)
) / 1000
| extend required_speed = geo_distance_km / time_diff_hours
| where required_speed > MaxKmPerHour
| project user_id, from_geo = prev_geo, to_geo = geo_location,
          distance_km = geo_distance_km, time_hours = time_diff_hours,
          impossible_speed = required_speed
```

---

## 9. Multi-Email Free-Tier Churn Detection

```kql
// Detect users cycling emails from same device to farm free tier
SecurityEvents
| where event_type == "signup"
| summarize emails = make_set(email), account_count = dcount(user_id) by device_id
| where account_count >= 5  // >=5 accounts from same device
| project device_id, account_count, emails,
          likely_free_tier_abuse = (account_count >= 5)
```

---

## 10. Forecasting & Trend Analysis

**KQL has built-in forecasting for capacity planning**

```kql
// Forecast next 7 days of signup volume
SecurityEvents
| where event_type == "signup"
| make-series signup_count = count() default=0 on timestamp step 1d
| extend forecast = series_decompose_forecast(signup_count, 7)  // 7 days ahead
| mv-expand timestamp to typeof(datetime), signup_count to typeof(long), forecast to typeof(double)
| project timestamp, actual = signup_count, predicted = forecast
```

---

## Production Migration Notes

### Data Ingestion Pipeline

```
Azure Event Hubs → ADX (Kusto) Ingestion
   ↓
SecurityEvents table (hot cache: 7-30 days)
   ↓
Cold storage: ADLS Gen2 (historical archive)
```

### ADX Table Schema

```kql
.create table SecurityEvents (
    timestamp: datetime,
    user_id: string,
    email: string,
    device_id: string,
    ip_address: string,
    session_id: string,
    event_type: string,
    anomaly_type: string,
    geo_location: string,
    metadata: dynamic  // JSON metadata
)

.create table SecurityEvents ingestion json mapping 'SecurityEventsMapping'
'[
    {"column":"timestamp", "path":"$.timestamp", "datatype":"datetime"},
    {"column":"user_id", "path":"$.user_id"},
    {"column":"email", "path":"$.email"},
    {"column":"device_id", "path":"$.device_id"},
    {"column":"ip_address", "path":"$.ip_address"},
    {"column":"session_id", "path":"$.session_id"},
    {"column":"event_type", "path":"$.event_type"},
    {"column":"anomaly_type", "path":"$.anomaly_type"},
    {"column":"geo_location", "path":"$.geo_location"},
    {"column":"metadata", "path":"$", "datatype":"dynamic"}
]'
```

### Partitioning & Performance

```kql
// Partition by date for time-series queries
.alter table SecurityEvents policy partitioning
```json
{
  "PartitionKeys": [
    {
      "ColumnName": "timestamp",
      "Kind": "UniformRange",
      "Properties": {
        "RangeSize": "1.00:00:00"  // 1 day partitions
      }
    }
  ]
}
```
```

---

## Secure OpenTelemetry Collector Configuration

**Why:** Production systems should ingest logs/metrics/traces via OpenTelemetry for vendor-neutral observability.

**Security concerns:** DoS protection, rate limiting, PII scrubbing.

**Source:** [OpenTelemetry Collector security best practices](https://opentelemetry.io/docs/collector/security/)

### Minimal Secure Collector Config

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
        max_recv_msg_size_mib: 16  # Limit message size
        read_buffer_size: 524288
      http:
        endpoint: 0.0.0.0:4318

processors:
  # Rate limiting to prevent DoS
  batch:
    timeout: 10s
    send_batch_size: 1024
    send_batch_max_size: 2048

  # PII scrubbing (redact sensitive fields)
  attributes:
    actions:
      - key: email
        action: delete  # Or hash/redact
      - key: ip_address
        action: hash
      - key: credit_card
        pattern: '\d{4}-\d{4}-\d{4}-\d{4}'
        action: delete

  # Resource limits
  memory_limiter:
    check_interval: 1s
    limit_mib: 512

exporters:
  # Azure Monitor
  azuremonitor:
    instrumentation_key: ${AZURE_MONITOR_KEY}
    endpoint: https://dc.services.visualstudio.com/v2/track

  # Event Hub (for ADX ingestion)
  azureeventhub:
    connection_string: ${EVENT_HUB_CONNECTION_STRING}
    partition_key: timestamp

service:
  pipelines:
    logs:
      receivers: [otlp]
      processors: [memory_limiter, attributes, batch]
      exporters: [azuremonitor, azureeventhub]
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [azuremonitor]
```

### PII Scrubbing Guidelines

1. **Hash** identifiers: emails, IPs, device IDs (use SHA-256 with salt)
2. **Delete** sensitive fields: credit cards, SSNs, passwords
3. **Truncate** geo data: city-level OK, GPS coordinates not OK
4. **Redact** user-generated content: messages, notes

---

## References

- [KQL Anomaly Detection](https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/anomaly-detection)
- [KQL RCA Functions](https://learn.microsoft.com/en-us/azure/data-explorer/kusto/query/root-cause-analysis)
- [OpenTelemetry Security](https://opentelemetry.io/docs/collector/security/)
- [ADX Time-Series Best Practices](https://learn.microsoft.com/en-us/azure/data-explorer/time-series-analysis)
