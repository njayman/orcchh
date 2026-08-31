from __future__ import annotations

import os

from fastapi import FastAPI


def setup_tracing(service_name: str, app: FastAPI) -> None:
    # Opt-in: only wired up when OTEL_EXPORTER_OTLP_ENDPOINT is set (the GKE+Jaeger
    # deployment sets it; local dev, self-checks, and the Docker/GCP path don't, so
    # they're unaffected). BatchSpanProcessor already degrades gracefully if the
    # collector is unreachable -- spans get dropped, not raised as request errors.
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()


def demo() -> None:
    # Neither branch should raise: unset must no-op cleanly, and set must wire up
    # instrumentation even though nothing is actually listening at that address
    # (span export is async/batched, so a dead collector doesn't surface here).
    os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
    setup_tracing("demo-service", FastAPI())

    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://127.0.0.1:4317"
    try:
        setup_tracing("demo-service", FastAPI())
    finally:
        del os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]

    print("tracing self-check passed")


if __name__ == "__main__":
    demo()
