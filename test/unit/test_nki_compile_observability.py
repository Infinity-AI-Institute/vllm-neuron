"""Focused observability tests for the NKI parser-to-tracer path."""

import json
from dataclasses import dataclass

import pytest
from _nki_test_loader import load_nki_modules

nki_cache, nki_compile, _ = load_nki_modules()


class ParserFrontend:
    def __init__(self, enable_backend_opt=False):
        self.enable_backend_opt = enable_backend_opt


class TracerFrontend(ParserFrontend):
    pass


class _Config:
    def __init__(self):
        self.backend_config_b64 = b"e30="
        self.output_specs = ()
        self.operand_output_aliases = {}


class _Nir:
    def build_config(self):
        return _Config()


@dataclass
class _CompileKernel:
    func: object
    lnc: int
    target: object
    _frontend_cls: type = ParserFrontend
    _enable_backend_opt: bool = False

    def _compile_opts(self):
        return object()

    def _cached_compile_to_bir(self, *, frontend, inputs, compile_opts):
        del inputs, compile_opts
        if type(frontend) is ParserFrontend:
            raise RuntimeError("inner function definitions are unsupported")
        return _Nir()


def test_parser_rejection_and_tracer_fallback_emit_structured_timings(
    monkeypatch, caplog
):
    monkeypatch.setattr(nki_compile, "CompileKernel", _CompileKernel)
    monkeypatch.setattr(nki_compile, "get_platform_target", lambda: "trn2")
    monkeypatch.setattr(nki_cache, "create_nki_cache_key", lambda *_args: "packed-key")
    monkeypatch.setattr(
        nki_cache, "compile_with_cache", lambda _key, compile_fn: compile_fn()
    )

    import nki.compiler.frontend

    monkeypatch.setattr(nki.compiler.frontend, "TracerFrontend", TracerFrontend)
    caplog.set_level("INFO", logger="vllm_neuron.nki.nki_compile")

    result = nki_compile.compile_nki(lambda: None, {}, (1,))

    assert result.dumped_config == "e30="
    events = [
        json.loads(record.getMessage().split("NKI_COMPILE_EVENT ", 1)[1])
        for record in caplog.records
        if "NKI_COMPILE_EVENT " in record.getMessage()
    ]
    parser_event = next(
        event
        for event in events
        if event["event"] == "frontend_compile"
        and event["frontend"] == "ParserFrontend"
    )
    tracer_event = next(
        event
        for event in events
        if event["event"] == "frontend_compile"
        and event["frontend"] == "TracerFrontend"
    )
    fallback_event = next(
        event for event in events if event["event"] == "tracer_fallback"
    )
    assert parser_event["outcome"] == "error"
    assert parser_event["error_type"] == "RuntimeError"
    assert tracer_event["outcome"] == "success"
    assert fallback_event["outcome"] == "success"
    assert fallback_event["reason"] == "parser_inner_function_definitions"
    assert all(event["cache_key"] == "packed-key" for event in events)
    assert all(isinstance(event["pid"], int) for event in events)
    assert all(event["duration_ms"] >= 0 for event in events)


def test_unrelated_parser_error_remains_fatal(monkeypatch, caplog):
    class OtherFailureKernel(_CompileKernel):
        def _cached_compile_to_bir(self, *, frontend, inputs, compile_opts):
            del frontend, inputs, compile_opts
            raise RuntimeError("unrelated parser failure")

    monkeypatch.setattr(nki_compile, "CompileKernel", OtherFailureKernel)
    monkeypatch.setattr(nki_compile, "get_platform_target", lambda: "trn2")
    monkeypatch.setattr(nki_cache, "create_nki_cache_key", lambda *_args: "fatal-key")
    monkeypatch.setattr(
        nki_cache, "compile_with_cache", lambda _key, compile_fn: compile_fn()
    )
    caplog.set_level("INFO", logger="vllm_neuron.nki.nki_compile")

    with pytest.raises(RuntimeError, match="unrelated parser failure"):
        nki_compile.compile_nki(lambda: None, {}, (1,))

    assert not any(
        '"event":"tracer_fallback"' in record.getMessage() for record in caplog.records
    )
