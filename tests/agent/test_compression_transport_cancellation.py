"""A timed-out compression must release its real HTTP stream and session lease."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock

import pytest
from openai import OpenAI

from agent import auxiliary_client as aux
from agent.conversation_compression import CompressionCommitFence, compress_context
from hermes_state import SessionDB
from run_agent import AIAgent


@pytest.mark.parametrize("phase", ["silent", "streaming", "late_headers"])
@pytest.mark.parametrize("cancel_source", ["timeout", "hard_stop"])
def test_compression_cancel_disconnects_only_its_request(tmp_path, phase, cancel_source):
    entered = threading.Event()
    disconnected = threading.Event()
    release_headers = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not body.get("stream"):
                payload = json.dumps({
                    "id": "sibling", "object": "chat.completion", "created": 0,
                    "model": "test", "choices": [{"index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop"}],
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            entered.set()
            if phase == "late_headers":
                release_headers.wait(5)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if phase == "streaming":
                chunk = {"id": "owner", "object": "chat.completion.chunk",
                         "created": 0, "model": "test", "choices": [{"index": 0,
                         "delta": {"content": "partial"}, "finish_reason": None}]}
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.flush()
            self.connection.settimeout(5)
            try:
                if self.connection.recv(1) == b"":
                    disconnected.set()
            except ConnectionResetError:
                disconnected.set()
            finally:
                self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    client = OpenAI(api_key="test", base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    max_retries=0, timeout=10)
    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id="owner", source="cli", model="test")
    agent = AIAgent(api_key="test", base_url=str(client.base_url), provider="custom",
                    model="test", quiet_mode=True, session_db=db, session_id="owner",
                    skip_context_files=True, skip_memory=True)
    agent._compression_feasibility_checked = True
    agent._cached_system_prompt = "unchanged"
    compressor = MagicMock()
    compressor.compression_count = 0
    agent.context_compressor = compressor
    fence = CompressionCommitFence()
    messages = [{"role": "user", "content": "keep this"}]
    result = {}

    def generate(*_args, **_kwargs):
        return aux._relay_sync_completion(
            client, {"model": "test", "messages": messages, "timeout": 10},
            create=lambda kwargs: aux._create_with_progress(client, kwargs, "compression"),
        )

    compressor.compress.side_effect = generate

    def run():
        try:
            result["value"] = compress_context(agent, messages, "unchanged", commit_fence=fence)
        except BaseException as exc:
            result["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    try:
        worker.start()
        assert entered.wait(5), f"compression never reached the HTTP server: {result!r}"
        if cancel_source == "timeout":
            assert fence.cancel_before_commit()
            fence.release_cancelled_compression_lock()
        else:
            agent._hard_interrupt_requested.set()
        worker.join(3)
        assert not worker.is_alive(), "cancelled compression worker did not return"
        release_headers.set()
        assert disconnected.wait(3), "provider kept generating after compression cancellation"
        assert "error" not in result, result.get("error")
        assert result["value"][0] == messages
        assert agent.session_id == "owner"
        assert db.try_acquire_compression_lock("owner", "next", ttl_seconds=30)
        assert not client.is_closed()
        sibling = client.chat.completions.create(model="test", messages=messages)
        assert sibling.choices[0].message.content == "ok"
    finally:
        release_headers.set()
        client.close()
        db.close()
        server.shutdown()
        server.server_close()
        server_thread.join(3)


def test_cancellation_does_not_shutdown_shared_http2_socket():
    from types import SimpleNamespace

    sock = MagicMock()
    chunks = SimpleNamespace(response=SimpleNamespace(extensions={
        "http_version": b"HTTP/2", "network_stream": MagicMock(),
    }))
    chunks.response.extensions["network_stream"].get_extra_info.return_value = sock
    with aux.aux_interrupt_protection(cancel_check=lambda: True):
        with pytest.raises(aux.AuxiliaryExplicitCancellation):
            with aux._cancel_chat_stream_on_host_stop(chunks):
                pytest.fail("cancelled stream was consumed")
    sock.shutdown.assert_not_called()
