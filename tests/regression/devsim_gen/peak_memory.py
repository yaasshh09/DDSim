from __future__ import annotations
import ctypes
import sys
from ctypes import wintypes


class _Counters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_mib() -> float:
    if sys.platform == "win32":
        c = _Counters()
        c.cb = ctypes.sizeof(c)
        k = ctypes.WinDLL("psapi")
        k.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
        h = ctypes.windll.kernel32.GetCurrentProcess()
        if not k.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return float(c.PeakWorkingSetSize) / 2**20
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10)


if __name__ == "__main__":
    before = peak_mib()
    block = bytearray(200 * 2**20)
    block[::4096] = b"x" * len(block[::4096])
    after = peak_mib()
    assert 190 < after - before < 260, (before, after)
    print(f"peak grew {after - before:.0f} MiB for a 200 MiB block")
