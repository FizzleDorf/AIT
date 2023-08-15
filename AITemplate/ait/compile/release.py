import os
import lzma
import hashlib
import json
import sys

def sha256sum(filename):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()

def filesize(filename):
    return os.stat(filename).st_size

def compress_file(filename):
    with open(filename, "rb") as f:
        data = f.read()
    with lzma.open(filename + ".xz", "wb", preset=9) as f:
        f.write(data)
    sha256 = sha256sum(filename + ".xz")
    return sha256, filesize(filename + ".xz")

def process_file(filename):
    file_size = filesize(filename)
    sha256 = sha256sum(filename)
    sha256_xz, file_size_xz = compress_file(filename)
    return sha256, file_size, sha256_xz, file_size_xz

def process(work_dir, model_name, dll_name, arch, height, width, batch_size, vram, out_dir, sd, model_type):
    path = os.path.join(work_dir, model_name)
    dll_path = os.path.join(path, dll_name)
    sha256, file_size, sha256_xz, file_size_xz = process_file(dll_path)
    _os = "windows" if sys.platform == "win32" else "linux"
    cuda = f"sm{arch}"
    if height is None or width is None:
        _reso = None
    else:
        _reso = max(height, width)
    _bs = batch_size
    compressed_name = f"{dll_name}.xz"
    compressed_path = os.path.join(path, compressed_name)
    subpath = f"{_os}/{cuda}/"
    if _reso is not None:
        subpath = subpath + f"bs{_bs}/{_reso}/"
    key = (subpath + compressed_name).replace("\\", "/")
    subpath = os.path.join(out_dir, subpath)
    os.makedirs(subpath, exist_ok=True)
    out_path = os.path.join(subpath, compressed_name)
    os.rename(compressed_path, out_path)
    data = {
        "os": _os,
        "cuda": cuda,
        "model": model_type,
        "sd": sd,
        "batch_size": _bs,
        "resolution": _reso,
        "vram": vram,
        "url": key,
        "compressed_size": file_size_xz,
        "size": file_size,
        "compressed_sha256": sha256_xz,
        "sha256": sha256,
    }
    if not os.path.exists(os.path.join(out_dir, "modules.json")):
        with open(os.path.join(out_dir, "modules.json"), "w") as f:
            json.dump({}, f)
    with open(os.path.join(out_dir, "modules.json"), "r") as f:
        modules = json.load(f)
        modules[key.replace("/", "_")] = data
    with open(os.path.join(out_dir, "modules.json"), "w") as f:
        json.dump(modules, f)

