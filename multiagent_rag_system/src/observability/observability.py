"""
OpenTelemetry tracing + LangSmith setup + @traced decorator.
"""
from __future__ import annotations
import functools
import time
from typing import Any, Callable, Optional

from ..utils.config_loader import get_settings
from ..logger import GLOBAL_LOGGER as logger

#Cached tracer instance

_tracer = None

def _get_tracer(name: str = "rag"):
    """Return cached tracer, or None if OTel is not configured."""
    global _tracer
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace
        if not get_settings().observability.otel.enabled:
            _tracer = None
            return None
        _tracer = trace.get_tracer(name)
        return _tracer
    except ImportError:
        _tracer = None
        return None


#OTel setup

def setup_otel() -> None:
    """Configure the OTel SDK and OTLP gRPC exporter from settings."""
    # Read settings inside the function — avoids stale module-level evaluation
    cfg = get_settings().observability.otel

    if not cfg.enabled:
        logger.info("otel_disabled")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({
            "service.name":    cfg.service_name,
            "service.version": cfg.service_version,
        })

        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(
            endpoint=str(cfg.endpoint),
            insecure=not str(cfg.endpoint).startswith("https"),  # TLS only for https
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        logger.info("otel_configured",
                    endpoint=str(cfg.endpoint),
                    service=cfg.service_name)

    except ImportError:
        logger.error("otel_packages_missing",
                       message="pip install opentelemetry-sdk opentelemetry-exporter-otlp")


#LangSmith setup
def setup_langsmith() -> None:
    """
    Validate LangSmith config at startup.
    Env vars should already be set in .env — this just confirms they're present
    and logs the active project.
    """
    import os
    cfg = get_settings().observability.langsmith

    if not cfg.enabled:
        logger.info("langsmith_disabled")
        return

    if not cfg.api_key:
        logger.info("langsmith_enabled_but_no_key",
                       message="Set LANGSMITH_API_KEY or disable LangSmith")
        return

    # Set env vars only if not already present — respects external configuration
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", cfg.api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", cfg.project)
    os.environ.setdefault("LANGCHAIN_ENDPOINT", str(cfg.endpoint))

    logger.info("langsmith_configured", project=cfg.project)


#Combined setup

def setup_observability() -> None:
    """
    Call once at application startup (inside lifespan).
    Order matters: LangSmith env vars must be set before LangGraph imports.
    """
    setup_langsmith()
    setup_otel()


#@traced decorator
def traced(span_name: str, attributes: Optional[dict] = None) -> Callable:
    """
    Decorator that wraps async OR sync functions in an OTel span.
    Falls back silently when OTel is not configured.

    Usage:
        @traced("retriever.retrieve", attributes={"agent": "retrieval"})
        async def retrieve(self, queries): ...

        @traced("utils.parse")
        def parse(self, text): ...
    """
    def decorator(fn: Callable) -> Callable:

        if not _is_async(fn):
            #Sync path
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = _get_tracer()
                if tracer is None:
                    return fn(*args, **kwargs)

                with tracer.start_as_current_span(span_name) as span:
                    _set_base_attributes(span, fn, attributes)
                    t0 = time.perf_counter()
                    try:
                        result = fn(*args, **kwargs)
                        span.set_attribute("status", "ok")
                        span.set_attribute("latency_ms", round((time.perf_counter() - t0) * 1000, 2))
                        return result
                    except Exception as exc:
                        _record_error(span, exc)
                        raise
            return sync_wrapper

        else:
            # ── Async path ────────────────────────────────────────────────
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = _get_tracer()
                if tracer is None:
                    return await fn(*args, **kwargs)

                with tracer.start_as_current_span(span_name) as span:
                    _set_base_attributes(span, fn, attributes)
                    t0 = time.perf_counter()
                    try:
                        result = await fn(*args, **kwargs)
                        span.set_attribute("status", "ok")
                        span.set_attribute("latency_ms", round((time.perf_counter() - t0) * 1000, 2))
                        return result
                    except Exception as exc:
                        _record_error(span, exc)
                        raise
            return async_wrapper

    return decorator


# ── Helpers ───────────────────────────────────────────────────────────────

def _is_async(fn: Callable) -> bool:
    import inspect
    return inspect.iscoroutinefunction(fn)


def _set_base_attributes(span: Any, fn: Callable, extra: Optional[dict]) -> None:
    span.set_attribute("function", fn.__qualname__)
    if extra:
        for k, v in extra.items():
            span.set_attribute(k, v)


def _record_error(span: Any, exc: Exception) -> None:
    span.set_attribute("status", "error")
    span.set_attribute("error.type", type(exc).__name__)
    span.set_attribute("error.message", str(exc))
    span.record_exception(exc)
    # Mark span status as ERROR per OTel spec
    try:
        from opentelemetry.trace import Status, StatusCode
        span.set_status(Status(StatusCode.ERROR, str(exc)))
    except ImportError:
        pass