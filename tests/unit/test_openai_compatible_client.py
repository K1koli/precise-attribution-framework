import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from precise_attribution.llm.openai_compatible import OpenAICompatibleClient


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"data": [{"id": "Qwen3.8-27B"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        assert payload["model"] == "Qwen3.8-27B"
        body = json.dumps({"choices": [{"message": {"content": "pong"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def test_openai_compatible_health_and_chat():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = OpenAICompatibleClient(base_url=f"http://127.0.0.1:{server.server_port}/v1", model="Qwen3.8-27B")
        assert client.health()["ok"] is True
        assert client.chat([{"role": "user", "content": "ping"}]) == "pong"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
