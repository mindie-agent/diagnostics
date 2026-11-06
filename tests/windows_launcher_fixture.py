"""Minimal PE header fixture for scheduler contracts; never executed."""
import struct


def write_launcher(path, *, subsystem=2, magic=0x20B):
    header = bytearray(64 + 24 + 70)
    header[:2] = b"MZ"
    struct.pack_into("<I", header, 60, 64)
    header[64:68] = b"PE\0\0"
    struct.pack_into("<H", header, 84, 70)
    struct.pack_into("<H", header, 88, magic)
    struct.pack_into("<H", header, 156, subsystem)
    path.write_bytes(header)
    return path
