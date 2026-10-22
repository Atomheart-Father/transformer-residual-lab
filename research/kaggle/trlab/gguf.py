"""纯 Python GGUF 读写（无第三方依赖）。

只读头部与张量表，不碰权重数据；合成 GGUF 用于 smoke test。
"""

from .errors import UnknownGGMLType
from .core import check_deadline

import struct
from pathlib import Path

def _base_consistency(scan_a, scan_b):
    """两份 base 是否来自同一 checkpoint：架构 + 张量名集合 + 每个张量形状 + 参数量。"""
    if not scan_a or not scan_b:
        return {"ok": None, "why": "有一侧没有 scan 结果"}
    if scan_a.get("architecture") != scan_b.get("architecture"):
        return {"ok": False,
                "why": f"architecture 不同：{scan_a.get('architecture')} "
                       f"vs {scan_b.get('architecture')}"}
    a = {t["name"]: tuple(t["dims"]) for t in scan_a.get("tensors", [])}
    b = {t["name"]: tuple(t["dims"]) for t in scan_b.get("tensors", [])}
    if set(a) != set(b):
        return {"ok": False,
                "why": f"张量名集合不同（只在 A: {sorted(set(a) - set(b))[:3]}；"
                       f"只在 B: {sorted(set(b) - set(a))[:3]}）"}
    bad = [n for n in a if a[n] != b[n]]
    if bad:
        n = bad[0]
        return {"ok": False,
                "why": f"{len(bad)} 个张量形状不同，例如 {n}: {a[n]} vs {b[n]}"}
    if scan_a.get("n_params") != scan_b.get("n_params"):
        return {"ok": False,
                "why": f"参数总数不同：{scan_a.get('n_params')} vs {scan_b.get('n_params')}"}
    return {"ok": True, "n_tensors": len(a), "architecture": scan_a.get("architecture"),
            "n_params": scan_a.get("n_params")}



# ---------------- 纯 Python GGUF 头解析（无第三方依赖） ----------------
# (name, block_size, type_size_bytes)，来自 ggml-common.h 的结构体实际大小
GGML_TYPES = {
    0:  ("F32",      1,   4),
    1:  ("F16",      1,   2),
    2:  ("Q4_0",     32,  18),
    3:  ("Q4_1",     32,  20),
    6:  ("Q5_0",     32,  22),
    7:  ("Q5_1",     32,  24),
    8:  ("Q8_0",     32,  34),
    9:  ("Q8_1",     32,  36),
    10: ("Q2_K",     256, 84),
    11: ("Q3_K",     256, 110),
    12: ("Q4_K",     256, 144),
    13: ("Q5_K",     256, 176),
    14: ("Q6_K",     256, 210),
    15: ("Q8_K",     256, 292),
    16: ("IQ2_XXS",  256, 66),
    17: ("IQ2_XS",   256, 74),
    18: ("IQ3_XXS",  256, 98),
    19: ("IQ1_S",    256, 50),
    20: ("IQ4_NL",   32,  18),
    21: ("IQ3_S",    256, 110),
    22: ("IQ2_S",    256, 82),
    23: ("IQ4_XS",   256, 136),
    24: ("I8",       1,   1),
    25: ("I16",      1,   2),
    26: ("I32",      1,   4),
    27: ("I64",      1,   8),
    28: ("F64",      1,   8),
    29: ("IQ1_M",    256, 56),
    30: ("BF16",     1,   2),
    34: ("TQ1_0",    256, 54),
    35: ("TQ2_0",    256, 66),
    36: ("MXFP4",    32,  17),
}



# 标量 value_type -> 字节数（GGUF 规范）
_GGUF_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}



def ggml_type_info(tid, strict=True):
    """P1-G：未知 type 必须 fail closed，不能按 8 bpw 假算。"""
    info = GGML_TYPES.get(tid)
    if info is None:
        if strict:
            raise UnknownGGMLType(
                f"未知 ggml type_id={tid}。拒绝按默认尺寸假算字节 —— "
                "该文件的字节拆分不可用。请升级 GGML_TYPES 表，或跳过此文件。")
        return ("UNKNOWN", None, None)
    return info



_GGML_NAME_TO_ID = {v[0]: k for k, v in GGML_TYPES.items()}



def _ggml_type_id(name):
    """格式名（如 "IQ2_XS"）→ ggml type_id；不认识返回 None。"""
    return _GGML_NAME_TO_ID.get(str(name).upper())



class _Reader:
    def __init__(self, f):
        self.f = f

    def read(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError(f"unexpected EOF: want {n}, got {len(b)}")
        return b

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        n = self.u64()
        return self.read(n).decode("utf-8", "replace")



def _skip_value(r, vt):
    """跳过任意 metadata value，不物化大数组（vocab 有 15 万个字符串）。"""
    if vt == 8:
        n = r.u64()
        r.f.seek(n, 1)
    elif vt == 9:
        et = r.u32()
        cnt = r.u64()
        if et in _GGUF_SCALAR:
            r.f.seek(_GGUF_SCALAR[et] * cnt, 1)
        elif et == 8:
            for _ in range(cnt):
                n = r.u64()
                r.f.seek(n, 1)
        else:
            raise ValueError(f"不支持的数组元素类型 {et}")
    elif vt in _GGUF_SCALAR:
        r.read(_GGUF_SCALAR[vt])
    else:
        raise ValueError(f"不支持的 value type {vt}")



def gguf_scan(path, deadline=None, strict_types=True):
    """读 GGUF 头，返回逐张量的类型/形状/真实字节。"""
    path = Path(path)
    with open(path, "rb") as f:
        r = _Reader(f)
        magic = r.read(4)
        if magic != b"GGUF":
            raise ValueError(f"不是 GGUF 文件（magic={magic!r}）: {path}")
        ver = r.u32()
        if ver not in (2, 3):
            raise ValueError(f"不支持的 GGUF 版本 {ver}")
        n_tensors = r.u64()
        n_kv = r.u64()

        WANT = {"general.architecture", "general.name", "general.file_type"}
        kv = {}
        for _ in range(n_kv):
            k = r.string()
            vt = r.u32()
            if k == "general.alignment" and vt == 4:
                kv[k] = r.u32()
            elif k in WANT and vt == 8:
                kv[k] = r.string()
            else:
                _skip_value(r, vt)

        align = int(kv.get("general.alignment", 32))
        tensors = []
        for i in range(n_tensors):
            if deadline and i % 64 == 0:
                check_deadline(deadline, "gguf_scan")
            name = r.string()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            tt = r.u32()
            off = r.u64()
            tensors.append({"name": name, "dims": dims, "type_id": tt, "offset": off})
        data_start = f.tell()

    data_start += (-data_start) % align

    per_type, per_class = {}, {}
    n_params, tensor_bytes = 0, 0
    for t in tensors:
        n = 1
        for d in t["dims"]:
            n *= d
        t["n_elements"] = n
        n_params += n

        tname, blk, tsz = ggml_type_info(t["type_id"], strict=strict_types)
        t["type"] = tname
        if blk is None:
            t["nbytes"] = None
            per_type[tname] = per_type.get(tname, 0) + 0
            continue
        nb = ((n + blk - 1) // blk) * tsz
        t["nbytes"] = nb
        tensor_bytes += nb
        per_type[tname] = per_type.get(tname, 0) + nb

        nm = t["name"].lower()
        if "token_embd" in nm or "output.weight" in nm:
            cls = "embed/head"
        elif "attn_" in nm:
            cls = "attn"
        elif "ffn_" in nm:
            cls = "ffn"
        else:
            cls = "other"
        per_class[cls] = per_class.get(cls, 0) + nb

    fsize = path.stat().st_size
    unknown_types = sorted({t["type"] for t in tensors if t["nbytes"] is None})
    return {
        "path": str(path),
        "file_bytes": fsize,
        "file_gb": round(fsize / 2**30, 3),
        "gguf_version": ver,
        "architecture": kv.get("general.architecture"),
        "alignment": align,
        "n_tensors": len(tensors),
        "n_params": n_params,
        "tensor_bytes": tensor_bytes,
        "header_bytes": fsize - tensor_bytes,
        # 有未知 type 时 tensor_bytes 只是下界，此时 bpw 是假数，必须给 None
        "bpw_effective": (round(tensor_bytes * 8 / n_params, 4)
                          if (n_params and not unknown_types) else None),
        "bpw_filesize": round(fsize * 8 / n_params, 4) if n_params else None,
        "bytes_per_type": dict(sorted(per_type.items(), key=lambda x: -x[1])),
        "bytes_per_class": dict(sorted(per_class.items(), key=lambda x: -x[1])),
        "unknown_types": unknown_types,
        "tensors": tensors,
    }



def gguf_write_synth(path, specs, alignment=32, strict_types=True):
    """写一个最小但合法的 GGUF v3，用于自检解析器。specs = [(name, dims, type_id)]

    strict_types=False 只给「故意构造未知 type 的坏文件」用：
    未知 type 没有块尺寸，这里塞占位字节让文件长度成立，
    扫描端仍会（也应该）拒绝用它算字节。
    """
    import io
    buf = io.BytesIO()

    def u32(x): buf.write(struct.pack("<I", x))
    def u64(x): buf.write(struct.pack("<Q", x))

    def s(x):
        b = x.encode()
        u64(len(b))
        buf.write(b)

    buf.write(b"GGUF")
    u32(3)
    u64(len(specs))
    u64(2)
    s("general.architecture"); u32(8); s("llama")
    s("general.alignment");    u32(4); u32(alignment)

    sizes, cur = [], 0
    for name, dims, tid in specs:
        s(name)
        u32(len(dims))
        for d in dims:
            u64(d)
        u32(tid)
        u64(cur)
        n = 1
        for d in dims:
            n *= d
        _, blk, tsz = ggml_type_info(tid, strict=strict_types)
        if blk is None:
            # 未知 type：占位尺寸，只为了把文件写出来。scan 端不会采用它。
            blk, tsz = 256, 64
        nb = ((n + blk - 1) // blk) * tsz
        sizes.append(nb)
        cur += ((nb + alignment - 1) // alignment) * alignment

    while len(buf.getvalue()) % alignment:
        buf.write(b"\x00")
    for nb in sizes:
        buf.write(b"\x00" * nb)

    Path(path).write_bytes(buf.getvalue())
    return Path(path)
