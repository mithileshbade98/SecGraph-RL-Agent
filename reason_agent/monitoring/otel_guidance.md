# OpenTelemetry Collector Security Guidance

This document provides secure configuration practices for the OpenTelemetry Collector in production.

**Why OpenTelemetry?** Vendor-neutral observability with logs, metrics, and traces.

**Security Concerns:**
- DoS attacks via excessive telemetry
- PII leakage in logs/traces
- Unauthorized access to collector endpoints

**Source:** [OpenTelemetry Collector Security Best Practices](https://opentelemetry.io/docs/collector/security/)

---

## 1. Secure Collector Configuration

### Minimal Secure Setup

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
        max_recv_msg_size_mib: 16  # Limit message size
        read_buffer_size: 524288    # 512KB
      http:
        endpoint: 0.0.0.0:4318

processors:
  # Rate limiting and memory protection
  memory_limiter:
    check_interval: 1s
    limit_mib: 512
    spike_limit_mib: 128

  # Batch to reduce load
  batch:
    timeout: 10s
    send_batch_size: 1024
    send_batch_max_size: 2048

  # PII scrubbing
  attributes:
    actions:
      - key: email
        action: delete
      - key: ip_address
        action: hash  # SHA-256 hash
      - key: device_id
        action: hash
      - key: credit_card
        pattern: '\d{4}-\d{4}-\d{4}-\d{4}'
        action: delete
      - key: ssn
        pattern: '\d{3}-\d{2}-\d{4}'
        action: delete

  # Filter sensitive spans
  filter:
    spans:
      exclude:
        match_type: strict
        attributes:
          - key: http.url
            value: "/api/secrets"

exporters:
  # Azure Monitor
  azuremonitor:
    instrumentation_key: ${AZURE_MONITOR_KEY}
    endpoint: https://dc.services.visualstudio.com/v2/track

  # Azure Event Hub (for ADX ingestion)
  azureeventhub:
    connection_string: ${EVENT_HUB_CONNECTION_STRING}
    partition_key: timestamp

service:
  pipelines:
    logs:
      receivers: [otlp]
      processors: [memory_limiter, attributes, batch]
      exporters: [azuremonitor, azureeventhub]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [azuremonitor]
    traces:
      receivers: [otlp]
      processors: [memory_limiter, filter, batch]
      exporters: [azuremonitor]

  # Telemetry for the collector itself
  telemetry:
    logs:
      level: info
    metrics:
      level: detailed
      address: 0.0.0.0:8888
```

---

## 2. DoS Protection Strategies

### Memory Limits

```yaml
processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 512         # Hard limit
    spike_limit_mib: 128   # Allow temporary spikes
```

### Rate Limiting (KEDA for Kubernetes)

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: otel-collector-scaler
spec:
  scaleTargetRef:
    name: otel-collector
  minReplicaCount: 2
  maxReplicaCount: 10
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: otelcol_receiver_accepted_spans
      threshold: "1000"
      query: rate(otelcol_receiver_accepted_spans[1m])
```

### Connection Limits

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        max_concurrent_streams: 100  # Limit concurrent streams
        keepalive:
          server_parameters:
            max_connection_idle: 5m
            max_connection_age: 30m
```

---

## 3. PII Scrubbing Best Practices

### Field-Level Scrubbing

```yaml
processors:
  attributes:
    actions:
      # Delete sensitive fields
      - key: password
        action: delete
      - key: auth_token
        action: delete

      # Hash identifiers (use consistent salt)
      - key: email
        action: hash
      - key: user_id
        action: hash
      - key: ip_address
        action: hash

      # Regex-based redaction
      - key: message
        action: extract
        pattern: '(email|phone):\s*(\S+)'
        replacement: '$1: [REDACTED]'

      # Truncate geo precision
      - key: geo.lat
        action: convert
        converter: truncate_float
        precision: 2  # City-level, not GPS
```

### Span Filtering

```yaml
processors:
  filter:
    spans:
      exclude:
        match_type: regexp
        attributes:
          - key: http.url
            value: '/api/(secrets|credentials|keys)/.*'
        span_names:
          - 'AuthService.Login'  # Filter auth spans
```

---

## 4. Authentication & Authorization

### Mutual TLS (mTLS)

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        tls:
          cert_file: /certs/server.crt
          key_file: /certs/server.key
          client_ca_file: /certs/ca.crt
          min_version: "1.3"
```

### API Key Authentication (Extension)

```yaml
extensions:
  bearertokenauth:
    token: ${OTEL_API_KEY}

receivers:
  otlp:
    protocols:
      grpc:
        auth:
          authenticator: bearertokenauth
```

---

## 5. Azure-Specific Security

### Managed Identity (Preferred)

```yaml
exporters:
  azuremonitor:
    # Use managed identity instead of instrumentation key
    connection_string: "InstrumentationKey=${MANAGED_IDENTITY_CLIENT_ID}@applicationinsights"
```

### Private Endpoints

```yaml
exporters:
  azureeventhub:
    # Use private link endpoint
    connection_string: "Endpoint=sb://secgraph-eh.privatelink.servicebus.windows.net/;..."
```

---

## 6. Monitoring the Collector

### Expose Metrics

```yaml
service:
  telemetry:
    metrics:
      address: 0.0.0.0:8888  # Prometheus scrape endpoint
```

### Key Metrics to Monitor

- `otelcol_receiver_refused_spans`: Dropped spans (DoS indicator)
- `otelcol_processor_batch_send_size`: Batch efficiency
- `otelcol_exporter_send_failed_spans`: Export failures
- `process_runtime_total_alloc_bytes`: Memory usage

### Alerts

```yaml
groups:
- name: otel-collector
  rules:
  - alert: OTelCollectorHighDropRate
    expr: rate(otelcol_receiver_refused_spans[1m]) > 100
    for: 5m
    annotations:
      summary: "High span drop rate detected"

  - alert: OTelCollectorMemoryHigh
    expr: process_runtime_total_alloc_bytes / 1024 / 1024 > 400
    for: 5m
    annotations:
      summary: "Collector memory usage > 400MB"
```

---

## 7. Production Checklist

- [ ] Memory limiter configured
- [ ] Rate limiting in place (KEDA or connection limits)
- [ ] PII scrubbing for all sensitive fields
- [ ] mTLS or API key authentication enabled
- [ ] Collector metrics exposed and monitored
- [ ] Logs at INFO level (not DEBUG in production)
- [ ] Secrets managed via env vars or Azure Key Vault
- [ ] Private endpoints for Azure services
- [ ] Alerts configured for drops/failures/memory
- [ ] Regular security audits of scrubbing rules

---

## 8. References

- [OpenTelemetry Collector Security](https://opentelemetry.io/docs/collector/security/)
- [Azure Monitor OpenTelemetry](https://learn.microsoft.com/en-us/azure/azure-monitor/app/opentelemetry-enable)
- [PII Scrubbing Processor](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/processor/attributesprocessor)
