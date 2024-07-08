# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import time
import logging
import http.server
import zipfile
from urllib.parse import parse_qs, urlparse
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_service, GenericService


class HttpFileServer(http.server.SimpleHTTPRequestHandler):
    web_dir = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.web_dir, **kwargs)


@register_service("microsoft/service/http_file_server")
class HttpFileServerService(GenericService):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section("microsoft/service/http_file_server")
        self.ip = config.getoption("ip", "0.0.0.0")
        self.port = config.getoption("port", 8000)
        self.name = config.getoption("processname", "core_http_server_service")
        self.web_dir = config.getoption("web_dir", None)
        assert self.web_dir is not None, "web_dir must be provided"

    def start(self, **kwargs):
        HttpFileServer.web_dir = self.web_dir
        self.httpd = http.server.HTTPServer((self.ip, self.port), HttpFileServer)
        self.httpd.serve_forever()

    def stop(self, **kwargs):
        pass

    def restart(self, **kwargs):
        pass
