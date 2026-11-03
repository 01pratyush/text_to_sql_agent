"""Optional tracing to a local Arize Phoenix server.

Turned on by PHOENIX_TRACING=1 (in .env or the environment). Start the server first:

    phoenix serve          # UI at http://localhost:6006

Then every agent run shows up in Phoenix as a trace: graph nodes, LLM calls
(prompts, outputs, tokens) and tool calls. Chat turns that share a thread_id are
grouped into one session, and each trace's metadata carries the log run_id, so
it matches the events in logs/agent.jsonl (filter in Phoenix with
metadata["run_id"] == "c01-r1").

Spans are exported in the background; if the server isn't running, the agent
still works, but the traces are lost and the exporter logs a warning and an
error after retrying for a few seconds.
Call setup_tracing() once, from an entry point, like setup_logging().
"""

import logging
import os

log = logging.getLogger("tracing")

DEFAULT_ENDPOINT = "http://localhost:6006"


def setup_tracing(project_name: str) -> None:
    """Send traces to Phoenix under project_name, if PHOENIX_TRACING is on."""
    if os.environ.get("PHOENIX_TRACING", "").lower() not in ("1", "true", "yes"):
        return
    # Plain OpenTelemetry rather than phoenix.otel.register: importing anything
    # under `phoenix` loads the whole Phoenix server package (~15 s, noisy logs).
    # Imported here so the agent runs without these packages when tracing is off
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.semconv.resource import ResourceAttributes
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")
    # Phoenix files traces under the project named in this resource attribute
    provider = TracerProvider(resource=Resource({ResourceAttributes.PROJECT_NAME: project_name}))
    # Exports from a background thread, so a slow or stopped server doesn't slow the agent
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")))
    # Hooks into LangChain's callback system, like LoggingCallbackHandler does
    LangChainInstrumentor().instrument(tracer_provider=provider)
    log.info("tracing_enabled", extra={"project": project_name, "endpoint": endpoint})
