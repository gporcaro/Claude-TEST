"""Diagnostic tools: ping, DNS, disk usage, service status."""

from __future__ import annotations

import asyncio
import re
import shutil
import socket

# Input sanitization: only allow safe hostnames/IPs
_SAFE_HOST_RE = re.compile(r"^[a-zA-Z0-9._-]+$")
_TIMEOUT = 10  # seconds


def _validate_host(host: str) -> str:
    """Validate and sanitize a hostname or IP."""
    host = host.strip()
    if not host or len(host) > 253:
        raise ValueError(f"Invalid host: {host!r}")
    if not _SAFE_HOST_RE.match(host):
        raise ValueError(f"Host contains invalid characters: {host!r}")
    return host


async def ping_host(host: str, count: int = 4, **_) -> dict:
    """Ping a host and return results."""
    host = _validate_host(host)
    count = min(max(1, count), 10)

    try:
        proc = await asyncio.create_subprocess_exec(
            "ping",
            "-c",
            str(count),
            "-W",
            "5",
            host,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT + count * 2)

        output = stdout.decode(errors="replace")
        if proc.returncode == 0:
            return {"status": "reachable", "host": host, "output": output}
        else:
            err_output = output or stderr.decode(errors="replace")
            return {"status": "unreachable", "host": host, "output": err_output}
    except asyncio.TimeoutError:
        return {"status": "timeout", "host": host, "error": "Ping timed out"}
    except Exception as e:
        return {"status": "error", "host": host, "error": str(e)}


async def dns_lookup(hostname: str, record_type: str = "A", **_) -> dict:
    """Perform DNS lookup."""
    hostname = _validate_host(hostname)
    record_type = record_type.upper()
    if record_type not in ("A", "AAAA", "MX", "CNAME", "TXT", "NS"):
        record_type = "A"

    try:
        # Use nslookup for cross-platform compatibility
        proc = await asyncio.create_subprocess_exec(
            "nslookup",
            f"-type={record_type}",
            hostname,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT)
        output = stdout.decode(errors="replace")

        # Also do a simple socket resolution for A records
        addresses = []
        if record_type == "A":
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, socket.getaddrinfo, hostname, None, socket.AF_INET
                )
                addresses = list({addr[4][0] for addr in result})
            except socket.gaierror:
                pass

        return {
            "hostname": hostname,
            "record_type": record_type,
            "addresses": addresses,
            "raw_output": output,
        }
    except asyncio.TimeoutError:
        return {"hostname": hostname, "error": "DNS lookup timed out"}
    except Exception as e:
        return {"hostname": hostname, "error": str(e)}


async def check_disk_usage(path: str = "/", **_) -> dict:
    """Check disk usage for a path."""
    # Sanitize path
    path = path.strip()
    if not path or ".." in path:
        path = "/"

    try:
        usage = await asyncio.get_event_loop().run_in_executor(None, shutil.disk_usage, path)
        total_gb = usage.total / (1024**3)
        used_gb = usage.used / (1024**3)
        free_gb = usage.free / (1024**3)
        percent = (usage.used / usage.total) * 100

        return {
            "path": path,
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "free_gb": round(free_gb, 2),
            "percent_used": round(percent, 1),
            "status": "critical" if percent > 90 else "warning" if percent > 80 else "ok",
        }
    except Exception as e:
        return {"path": path, "error": str(e)}


async def check_service_status(service_name: str, **_) -> dict:
    """Check if a service/process is running."""
    service_name = service_name.strip()
    if not service_name or not _SAFE_HOST_RE.match(service_name):
        return {"error": f"Invalid service name: {service_name!r}"}

    try:
        # Use pgrep for cross-platform process checking
        proc = await asyncio.create_subprocess_exec(
            "pgrep",
            "-f",
            service_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT)

        pids = stdout.decode().strip().split("\n") if stdout.decode().strip() else []

        if pids and pids[0]:
            return {
                "service": service_name,
                "status": "running",
                "pids": pids[:5],  # Cap at 5 PIDs
            }
        else:
            return {
                "service": service_name,
                "status": "not_running",
            }
    except asyncio.TimeoutError:
        return {"service": service_name, "error": "Check timed out"}
    except Exception as e:
        return {"service": service_name, "error": str(e)}
