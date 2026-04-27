"""TLS 1.3 context helpers for paho-mqtt clients.

Run a private CA in production. For development, a self-signed cert is fine
provided the device pins it.
"""
from __future__ import annotations

import ssl
from pathlib import Path
from typing import Optional


def build_tls_context(ca_cert: Optional[Path] = None,
                      client_cert: Optional[Path] = None,
                      client_key: Optional[Path] = None,
                      verify_peer: bool = True) -> ssl.SSLContext:
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH,
                                     cafile=str(ca_cert) if ca_cert else None)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    if client_cert and client_key:
        ctx.load_cert_chain(certfile=str(client_cert), keyfile=str(client_key))
    if not verify_peer:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx
