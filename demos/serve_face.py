"""
Face Expression Demo - 本地 HTTP 服务器
Local HTTP server for Face Expression Demo

启动方式 / Usage:
    python3 demos/serve_face.py

启动后访问 / Open in browser:
    http://localhost:8090
"""

import http.server
import os
import sys

PORT = 8090

# GLB 模型的原始路径 / Original path of GLB model
GLB_SOURCE = "/Users/johngao/Desktop/6d2ca87dcc5e64a9af5971235d683e2d.glb"

# HTML 文件路径 / HTML file path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(SCRIPT_DIR, "face_expression.html")


class FaceExpressionHandler(http.server.BaseHTTPRequestHandler):
    """自定义请求处理器 / Custom request handler"""

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_file(HTML_PATH, "text/html; charset=utf-8")
        elif self.path == "/model.glb":
            self._serve_large_file(GLB_SOURCE, "model/gltf-binary")
        else:
            self.send_error(404, "Not Found")

    def _serve_file(self, filepath, content_type):
        """提供普通文件 / Serve regular files"""
        try:
            with open(filepath, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(data))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_error(404, f"File not found: {filepath}")

    def _serve_large_file(self, filepath, content_type):
        """
        流式提供大文件（支持进度条）
        Stream large files (supports progress bar)
        """
        try:
            file_size = os.path.getsize(filepath)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", file_size)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()

            # 分块传输，避免内存压力 / Chunked transfer to reduce memory pressure
            chunk_size = 1024 * 1024  # 1MB per chunk
            with open(filepath, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except FileNotFoundError:
            self.send_error(404, f"GLB file not found: {filepath}")
        except BrokenPipeError:
            pass  # 客户端断开连接 / Client disconnected

    def log_message(self, format, *args):
        """简化日志格式 / Simplified log format"""
        msg = format % args
        # 不打印模型文件的传输日志（太长）/ Skip verbose model transfer logs
        if "/model.glb" not in msg or "200" in msg:
            sys.stderr.write(f"  [{self.log_date_time_string()}] {msg}\n")


def main():
    # 检查文件是否存在 / Check files exist
    if not os.path.exists(GLB_SOURCE):
        print(f"❌ GLB 文件不存在 / GLB file not found: {GLB_SOURCE}")
        sys.exit(1)
    if not os.path.exists(HTML_PATH):
        print(f"❌ HTML 文件不存在 / HTML file not found: {HTML_PATH}")
        sys.exit(1)

    glb_size_mb = os.path.getsize(GLB_SOURCE) / (1024 * 1024)
    print(f"╔══════════════════════════════════════════╗")
    print(f"║   Face Expression Demo Server            ║")
    print(f"╠══════════════════════════════════════════╣")
    print(f"║  GLB 模型: {glb_size_mb:.1f} MB                       ║")
    print(f"║  地址 / URL: http://localhost:{PORT}        ║")
    print(f"║  按 Ctrl+C 停止 / Press Ctrl+C to stop   ║")
    print(f"╚══════════════════════════════════════════╝")

    server = http.server.HTTPServer(("", PORT), FaceExpressionHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止 / Server stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
