"""
OpenTelemetry tracing + LangSmith setup + @traced decorator.
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable

from ..utils.config_loader import get_settings
from ..logger.logger import GLOBAL_LOGGER as logger

logger = logging.getLogger("rag.observability")

settings = get_settings()

def setup_otel(cfg= settings.observability, service_name: str = "multi-agent-rag") -> None:
    """Configure the OTEL SDK and OTLP gRPC exporter."""
    if not cfg.otel_enabled:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=cfg.otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        logger.info(f"OTEL tracing → {cfg.otlp_endpoint}")

    except ImportError:
        logger.warning("opentelemetry packages not installed — tracing disabled")


def setup_langsmith(cfg = settings.observability, api_key = settings.langsmith_api_key, project: str = "multi-agent-rag") -> None:
    """Enable LangSmith tracing via environment variables."""
    if not cfg.langsmith_enabled or not api_key:
        return
    import os
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]    = api_key
    os.environ["LANGCHAIN_PROJECT"]    = project
    logger.info(f"LangSmith tracing → project={project}")


def _get_tracer(name: str = "rag"):
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except ImportError:
        return None


def traced(span_name: str) -> Callable:
    """
    Decorator that wraps an async method in an OpenTelemetry span.
    Falls back silently when OTLP is not configured.

    Usage:
        @traced("retriever.retrieve")
        async def retrieve(self, queries): ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = _get_tracer()
            if tracer is None:
                return await fn(*args, **kwargs)

            with tracer.start_as_current_span(span_name) as span:
                t0 = time.perf_counter()
                try:
                    result = await fn(*args, **kwargs)
                    span.set_attribute("status", "ok")
                    span.set_attribute("latency_ms", round((time.perf_counter() - t0) * 1000, 2))
                    return result
                except Exception as exc:
                    span.set_attribute("status", "error")
                    span.set_attribute("error", str(exc))
                    span.record_exception(exc)
                    raise
        return wrapper
    return decorator