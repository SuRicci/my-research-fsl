"""Read selected members of a public ZIP with HTTP ranges and CRC verification."""
from __future__ import annotations
import io
import struct
import zlib
import zipfile
from pathlib import Path
import requests
from .records import sha256


def byte_range(url, start, end):
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(15, 60))
            r.raise_for_status()
            if r.status_code != 206 or not r.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                raise IOError("Range not honored")
            if len(r.content) != end-start+1:
                raise IOError("Incomplete range")
            return r.content
        except Exception:
            if attempt == 2:
                raise


class RemoteZipReader(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        r = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=(15,30))
        if r.status_code != 206:
            r.close()
            raise IOError("Archive does not support public range access")
        self.size = int(r.headers["Content-Range"].split("/")[-1])
        self.etag = r.headers.get("ETag")
        r.close()

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0: raise ValueError("Negative seek")
        return self.pos

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.size,self.pos+n)
        if end <= self.pos: return b""
        data = byte_range(self.url,self.pos,end-1)
        self.pos = end
        return data


def read_member(reader, info, dest):
    dest = Path(dest)
    if dest.exists():
        data = dest.read_bytes()
    else:
        end = min(reader.size-1, info.header_offset + 30 + len(info.filename.encode("utf-8")) + len(info.extra) + info.compress_size + 1024)
        blob = byte_range(reader.url,info.header_offset,end)
        header = struct.unpack("<4s5H3I2H",blob[:30])
        if header[0] != b"PK\x03\x04": raise ValueError("ZIP header mismatch")
        offset = 30 + header[-2] + header[-1]
        if len(blob) < offset + info.compress_size:
            blob = byte_range(reader.url,info.header_offset,info.header_offset+offset+info.compress_size-1)
        compressed = blob[offset:offset+info.compress_size]
        if info.compress_type == zipfile.ZIP_DEFLATED: data = zlib.decompress(compressed,-15)
        elif info.compress_type == zipfile.ZIP_STORED: data = compressed
        else: raise ValueError("Unsupported compression")
    if len(data) != info.file_size or zlib.crc32(data)&0xffffffff != info.CRC:
        raise ValueError(f"Archive member CRC/length mismatch: {info.filename}")
    if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True)
        temp = dest.with_suffix(".download")
        temp.write_bytes(data)
        temp.replace(dest)
    return {"file":dest.name,"archive_member":info.filename,"bytes":len(data),"crc32":f"{info.CRC:08x}","sha256":sha256(dest)}
