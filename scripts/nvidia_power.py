"""Read the X driver's actual PowerMizer preference, without changing it."""

import ctypes as c
import json
import subprocess
import threading


_lock = threading.Lock()


def read_power_modes():
    # NV-CONTROL GPU target=1, PowerMizer=334, GPU UUID string=52.
    # Newer nvidia-settings NVML readback can disagree with the X driver.
    with _lock:
        x = c.CDLL("libX11.so.6")
        nv = c.CDLL("libXNVCtrl.so.0")
        x.XOpenDisplay.argtypes = [c.c_char_p]
        x.XOpenDisplay.restype = c.c_void_p
        x.XCloseDisplay.argtypes = [c.c_void_p]
        x.XFree.argtypes = [c.c_void_p]
        nv.XNVCTRLQueryTargetCount.argtypes = [c.c_void_p, c.c_int, c.POINTER(c.c_int)]
        nv.XNVCTRLQueryTargetCount.restype = c.c_int
        nv.XNVCTRLQueryTargetAttribute.argtypes = [
            c.c_void_p, c.c_int, c.c_int, c.c_uint, c.c_uint, c.POINTER(c.c_int),
        ]
        nv.XNVCTRLQueryTargetAttribute.restype = c.c_int
        nv.XNVCTRLQueryTargetStringAttribute.argtypes = [
            c.c_void_p, c.c_int, c.c_int, c.c_uint, c.c_uint, c.POINTER(c.c_void_p),
        ]
        nv.XNVCTRLQueryTargetStringAttribute.restype = c.c_int
        display = x.XOpenDisplay(None)
        if not display:
            raise RuntimeError("Cannot verify PowerMizer: current X display is unavailable")
        try:
            count = c.c_int()
            if not nv.XNVCTRLQueryTargetCount(display, 1, c.byref(count)) or count.value <= 0:
                raise RuntimeError("Cannot enumerate NV-CONTROL GPUs")
            modes = []
            for gpu in range(count.value):
                mode, uuid = c.c_int(), c.c_void_p()
                if not nv.XNVCTRLQueryTargetAttribute(display, 1, gpu, 0, 334, c.byref(mode)):
                    raise RuntimeError(f"Cannot query PowerMizer on NV-CONTROL GPU {gpu}")
                if not nv.XNVCTRLQueryTargetStringAttribute(display, 1, gpu, 0, 52, c.byref(uuid)):
                    raise RuntimeError(f"Cannot identify NV-CONTROL GPU {gpu}")
                if not uuid.value:
                    raise RuntimeError(f"Empty UUID on NV-CONTROL GPU {gpu}")
                try:
                    identity = c.string_at(uuid.value).decode("ascii")
                finally:
                    x.XFree(uuid)
                modes.append({"nvctrl_gpu": gpu, "uuid": identity, "power_mizer_mode": mode.value})
            return modes
        finally:
            x.XCloseDisplay(display)


def require_max_performance(modes, expected_uuids=None):
    uuids = [mode["uuid"] for mode in modes]
    if not modes or any(not uuid for uuid in uuids) or len(set(uuids)) != len(uuids):
        raise RuntimeError(f"Missing or duplicate PowerMizer GPU identities: {modes}")
    if expected_uuids is not None and set(uuids) != set(expected_uuids):
        raise RuntimeError(f"X and CUDA GPU coverage differs: {uuids} versus {expected_uuids}")
    if any(mode["power_mizer_mode"] != 1 for mode in modes):
        raise RuntimeError(f"Prefer maximum performance is not active on every GPU: {modes}")


def cuda_gpu_uuids():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=10,
    )
    uuids = [line.strip() for line in output.splitlines() if line.strip()]
    if not uuids:
        raise RuntimeError("No CUDA GPU UUIDs available")
    return uuids


if __name__ == "__main__":
    modes = read_power_modes()
    print(json.dumps(modes, indent=2), flush=True)
    require_max_performance(modes, cuda_gpu_uuids())
