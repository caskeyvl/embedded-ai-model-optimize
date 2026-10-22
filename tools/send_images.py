"""
send_images.py -- run your model on "the board": stream images to the
firmware, collect its predictions, report accuracy.

    # Renode (emulated board; launched for you, headless):
    python3 tools/send_images.py --pte submission/submission.pte --val-n 500 --jobs 4

    # The real board, firmware + model already flashed (flash_board.sh):
    python3 tools/send_images.py --port /dev/ttyUSB0 --val-n 500

    # An already-running Renode whose USART2 is a server socket:
    python3 tools/send_images.py --socket localhost:3456 --val-n 100

Images come from your validation split (data.py; the first --val-n images)
or from an .npz file with `images` (uint8 [N,32,32,3], RGB, HWC) and
`labels` (int [N]). This is exactly the harness the grader runs on its
hidden test images, with the grader's copy of the same firmware.

Add --config/--checkpoint to also compute the laptop's simulated int8
predictions for the same images (eval.py's int8 model) and report how
often the board agrees: a large gap means something differs between your
export and what you evaluated (see README.md "Laptop vs. board").

Protocol (firmware src/camera.h and src/main.cpp):
  host -> device:  A5 5A | idx:u16 | len:u16=3072 | RGB888 HWC payload | crc32:u32
                   (little-endian; crc32 = zlib.crc32 of idx|len|payload)
  device -> host:  BOOT,<fw>,<pte_len>,<planned>   INFO,...   READY
                   R,<idx>,<pred>,<cycles>,<status>
                   LOAD_ERR,<reason>[,<detail>]      FAULT,<cfsr>,<pc>
"""

import argparse
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from contextlib import contextmanager

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)          # the lab repo root (tools/ is inside it)
FIRMWARE_ELF = os.path.join(REPO, "firmware", "firmware.elf")
RENODE_REPL = os.path.join(REPO, "renode", "stm32f4_discovery_lab3.repl")
MODEL_ADDR = 0x08080000
FRAME_BYTES = 32 * 32 * 3

# Seconds. Renode needs ~10 s to start; the slowest legal int8 model takes
# a few seconds per image under Renode. An fp32 layer left unquantized runs
# on slow portable kernels, hence the generous per-image limit.
BOOT_TIMEOUT = 120
IMAGE_TIMEOUT = 120


class DeviceError(Exception):
    """The firmware reported LOAD_ERR or FAULT, or stopped responding."""

    def __init__(self, kind, line, log=None):
        super().__init__(f"{kind}: {line}")
        self.kind = kind          # LOAD_ERR | FAULT | TIMEOUT | DISCONNECTED
        self.line = line
        self.log = log or []


# ---------------------------------------------------------------- links ----

class Link:
    """Line-oriented link to the firmware: write bytes, read text lines."""

    def __init__(self):
        self._buf = b""
        self.log = []             # every line received, for error reports

    def write(self, data: bytes):
        raise NotImplementedError

    def _read_some(self, timeout: float) -> bytes:
        raise NotImplementedError

    def read_line(self, timeout: float):
        """Next non-empty line, or None on timeout."""
        end = time.time() + timeout
        while True:
            while b"\n" in self._buf:
                raw, self._buf = self._buf.split(b"\n", 1)
                line = raw.decode(errors="replace").strip()
                if line:
                    self.log.append(line)
                    return line
            left = end - time.time()
            if left <= 0:
                return None
            self._buf += self._read_some(min(left, 0.5))

    def close(self):
        pass


class SocketLink(Link):
    def __init__(self, sock: socket.socket):
        super().__init__()
        self.sock = sock

    def write(self, data):
        self.sock.sendall(data)

    def _read_some(self, timeout):
        self.sock.settimeout(timeout)
        try:
            data = self.sock.recv(4096)
        except socket.timeout:
            return b""
        if not data:
            raise DeviceError("DISCONNECTED", "connection closed", self.log)
        return data

    def close(self):
        self.sock.close()


class SerialLink(Link):
    def __init__(self, port, baud):
        super().__init__()
        import serial   # pyserial; only needed for the real board
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.ser.reset_input_buffer()

    def write(self, data):
        self.ser.write(data)
        self.ser.flush()

    def _read_some(self, timeout):
        self.ser.timeout = timeout
        return self.ser.read(max(1, self.ser.in_waiting))

    def close(self):
        self.ser.close()


def connect_socket(host, port, timeout=30.0) -> socket.socket:
    end = time.time() + timeout
    while True:
        try:
            return socket.create_connection((host, port), timeout=2)
        except OSError:
            if time.time() > end:
                raise
            time.sleep(0.2)


def reset_board() -> bool:
    """Reset a real board through its ST-LINK (OpenOCD), so the boot banner
    arrives after the serial port is open. Returns False if that failed."""
    if not shutil.which("openocd"):
        return False
    r = subprocess.run(["openocd", "-f", "interface/stlink.cfg", "-f", "target/stm32f4x.cfg",
                        "-c", "transport select hla_swd", "-c", "init; reset run; shutdown"],
                       capture_output=True, text=True)
    return r.returncode == 0


# --------------------------------------------------------------- Renode ----

def find_renode():
    for cand in (os.environ.get("RENODE"), shutil.which("renode"),
                 os.path.expanduser("~/renode_portable/renode")):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    sys.exit("Renode not found: put `renode` on PATH or set RENODE=/path/to/renode")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def renode_session(model_bin, firmware=FIRMWARE_ELF, repl=RENODE_REPL, log_path=None):
    """Start a headless Renode with the firmware and model loaded, connect to
    its USART2 socket, then start the CPU (so no boot output is missed).
    Yields a SocketLink; kills Renode on exit."""
    renode = find_renode()
    tmp = tempfile.mkdtemp(prefix="lab3_renode_")
    uart_port, mon_port = _free_port(), _free_port()
    resc = os.path.join(tmp, "lab3.resc")
    with open(resc, "w") as f:
        f.write(f"""mach create "lab3"
machine LoadPlatformDescription @{os.path.abspath(repl)}
sysbus LoadELF @{os.path.abspath(firmware)}
sysbus LoadBinary @{os.path.abspath(model_bin)} 0x{MODEL_ADDR:08X}
emulation CreateServerSocketTerminal {uart_port} "uart" false
connector Connect sysbus.usart2 uart
""")
    log = open(log_path or os.path.join(tmp, "renode.log"), "wb")
    proc = subprocess.Popen([renode, "--disable-gui", "--port", str(mon_port), "-e", f"include @{resc}"],
                            stdout=log, stderr=subprocess.STDOUT, cwd=tmp, start_new_session=True)
    link = mon = None
    try:
        try:
            link = SocketLink(connect_socket("127.0.0.1", uart_port, timeout=BOOT_TIMEOUT))
            mon = connect_socket("127.0.0.1", mon_port, timeout=30)
        except OSError:
            raise DeviceError("TIMEOUT", f"Renode did not open its sockets (log: {log.name})")
        time.sleep(0.5)
        mon.sendall(b"start\n")
        yield link
    finally:
        for s in (link, mon):
            try:
                s and s.close()
            except OSError:
                pass
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass
        proc.wait()
        log.close()
        if log_path is None:
            shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------- protocol ----

def encode_frame(idx: int, img: np.ndarray) -> bytes:
    payload = np.ascontiguousarray(img, dtype=np.uint8).tobytes()
    assert len(payload) == FRAME_BYTES, f"image must be 32x32x3 uint8 HWC, got {img.shape}"
    hdr = struct.pack("<HH", idx & 0xFFFF, FRAME_BYTES)
    return b"\xA5\x5A" + hdr + payload + struct.pack("<I", zlib.crc32(hdr + payload))


def wait_ready(link: Link, timeout=BOOT_TIMEOUT) -> dict:
    """Read the boot banner. Returns {"fw", "pte_len", "planned", "info"};
    raises DeviceError on LOAD_ERR, FAULT or timeout."""
    boot = {"info": {}}
    end = time.time() + timeout
    while True:
        line = link.read_line(max(0.0, end - time.time()))
        if line is None:
            raise DeviceError("TIMEOUT", "no READY from the firmware", link.log)
        if line.startswith("BOOT,"):
            parts = line.split(",")
            boot.update(fw=parts[1], pte_len=int(parts[2]), planned=int(parts[3]))
        elif line.startswith("INFO,"):
            parts = line.split(",")
            boot["info"][parts[1]] = parts[2:]
        elif line.startswith("LOAD_ERR"):
            raise DeviceError("LOAD_ERR", line, link.log)
        elif line.startswith("FAULT"):
            raise DeviceError("FAULT", line, link.log)
        elif line == "READY":
            return boot


def classify(link: Link, images, indices, timeout=IMAGE_TIMEOUT, progress=None) -> dict:
    """Send each image, wait for its R line. Returns {idx: (pred, cycles, status)}.
    A CRC/length error is retried once. Raises DeviceError on FAULT or a
    per-image timeout."""
    out = {}
    for img, idx in zip(images, indices):
        for attempt in range(2):
            link.write(encode_frame(idx, img))
            while True:
                line = link.read_line(timeout)
                if line is None:
                    raise DeviceError("TIMEOUT", f"no result for image {idx} within {timeout:.0f} s",
                                      link.log)
                if line.startswith("FAULT"):
                    raise DeviceError("FAULT", line, link.log)
                if line.startswith("R,"):
                    _, ridx, pred, cycles, status = line.split(",", 4)
                    if int(ridx) == (idx & 0xFFFF):
                        break
            if status not in ("CRC_ERR", "LEN_ERR", "TIMEOUT"):
                break
        out[idx] = (int(pred), int(cycles), status)
        if progress:
            progress()
    return out


def run_on_renode(model_bin, images, jobs=1, firmware=FIRMWARE_ELF, repl=RENODE_REPL,
                  log_dir=None, verbose=True):
    """Classify `images` (uint8 [N,32,32,3]) on `jobs` Renode instances in
    parallel (each instance is single-threaded; frames are independent).
    Returns (boot dict, {idx: (pred, cycles, status)}). Raises DeviceError."""
    n = len(images)
    jobs = max(1, min(jobs, n))
    shards = [list(range(i, n, jobs)) for i in range(jobs)]
    results, boots, errors = {}, [], []
    done = [0]
    lock = threading.Lock()
    t0 = time.time()

    def tick():
        with lock:
            done[0] += 1
            if verbose and (done[0] % 25 == 0 or done[0] == n):
                el = time.time() - t0
                print(f"  {done[0]}/{n} images  ({el:.0f} s, {el / done[0] * jobs:.1f} s/image/instance)",
                      flush=True)

    def worker(k):
        try:
            log_path = os.path.join(log_dir, f"renode_{k}.log") if log_dir else None
            with renode_session(model_bin, firmware, repl, log_path) as link:
                boot = wait_ready(link)
                with lock:
                    boots.append(boot)
                r = classify(link, images[shards[k]], shards[k], progress=tick)
                with lock:
                    results.update(r)
        except DeviceError as e:
            with lock:
                errors.append(e)
        except Exception as e:  # noqa: BLE001 -- report anything as a device error
            with lock:
                errors.append(DeviceError("HOST_ERROR", repr(e)))

    threads = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return boots[0], results


# ------------------------------------------------------------------ CLI ----

def load_images(args):
    if args.images:
        d = np.load(args.images)
        return d["images"].astype(np.uint8), d["labels"].astype(np.int64)
    sys.path.insert(0, HERE)
    from data import load_split
    _, (xva, yva) = load_split()
    return xva[:args.val_n].numpy(), yva[:args.val_n].numpy()


def summarize(preds, labels, boot, host_preds=None):
    from data import CLASS_NAMES
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    acc = float((preds == labels).mean())
    conf = np.zeros((10, 10), dtype=int)
    for p, y in zip(preds, labels):
        if 0 <= p < 10:
            conf[y, p] += 1
    print(f"\nfirmware {boot.get('fw')}: .pte {boot.get('pte_len'):,} bytes, "
          f"planned arena {boot.get('planned'):,} bytes")
    print(f"on-device accuracy: {acc:.4f}  ({int((preds == labels).sum())}/{len(labels)})")
    print("per class: " + "  ".join(f"{CLASS_NAMES[c][:5]} {conf[c, c] / max(1, conf[c].sum()):.2f}"
                                    for c in range(10)))
    summary = {"accuracy": acc, "n": int(len(labels)), "boot": boot,
               "per_class_accuracy": [float(conf[c, c] / max(1, conf[c].sum())) for c in range(10)],
               "confusion": conf.tolist()}
    if host_preds is not None:
        agree = float((np.asarray(host_preds) == preds).mean())
        host_acc = float((np.asarray(host_preds) == labels).mean())
        print(f"laptop simulated int8 accuracy: {host_acc:.4f}; board agrees with laptop on {agree:.4f}")
        summary.update(host_accuracy=host_acc, board_host_agreement=agree)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--port", help="serial port of the real board (e.g. /dev/ttyUSB0)")
    src.add_argument("--socket", help="HOST:PORT of a running Renode's USART2 socket")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--no-reset", action="store_true",
                    help="with --port: don't reset the board through OpenOCD; press its reset button instead")
    ap.add_argument("--pte", help="the .pte to run under Renode (default: submission/submission.pte)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel Renode instances")
    ap.add_argument("--val-n", type=int, default=200, help="first N validation images")
    ap.add_argument("--images", help=".npz with images (uint8 N,32,32,3) and labels")
    ap.add_argument("--config", help="with --checkpoint: also compare to the laptop's int8 simulation")
    ap.add_argument("--checkpoint")
    ap.add_argument("--out", help="write per-image results and the summary as JSON")
    ap.add_argument("--renode-logs", help="keep Renode's logs in this directory")
    args = ap.parse_args()

    images, labels = load_images(args)
    print(f"{len(images)} images")
    idx = list(range(len(images)))
    try:
        if args.port or args.socket:
            if args.port:
                link = SerialLink(args.port, args.baud)
                if args.no_reset or not reset_board():
                    print(f"Reset the board now (waiting up to {BOOT_TIMEOUT} s for READY) ...")
            else:
                host, port = args.socket.rsplit(":", 1)
                link = SocketLink(connect_socket(host, int(port)))
            boot = wait_ready(link)
            results = classify(link, images, idx, progress=None)
            link.close()
        else:
            pte = args.pte or os.path.join(REPO, "submission", "submission.pte")
            from pack_model import pack
            with open(pte, "rb") as f:
                blob = pack(f.read())
            tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
            tmp.write(blob)
            tmp.close()
            if args.renode_logs:
                os.makedirs(args.renode_logs, exist_ok=True)
            print(f"Renode: {pte} on {args.jobs} instance(s) ...")
            try:
                boot, results = run_on_renode(tmp.name, images, jobs=args.jobs,
                                              log_dir=args.renode_logs)
            finally:
                os.unlink(tmp.name)
    except DeviceError as e:
        print(f"\nDEVICE ERROR -- {e}")
        for line in e.log[-15:]:
            print(f"  | {line}")
        sys.exit(2)

    preds = [results[i][0] for i in idx]
    host_preds = None
    if args.config and args.checkpoint:
        import torch
        from data import normalize
        from export import add_quant_args, quantize_model
        from model import load_model
        qargs = add_quant_args(argparse.ArgumentParser()).parse_args([])
        q = quantize_model(load_model(args.config, args.checkpoint), qargs)
        with torch.no_grad():
            x = normalize(torch.from_numpy(images)).contiguous(memory_format=torch.channels_last)
            host_preds = q(x).argmax(1).tolist()
    summary = summarize(preds, labels, boot, host_preds)
    bad = [i for i in idx if results[i][2] != "OK"]
    if bad:
        print(f"WARNING: {len(bad)} images did not return OK, e.g. image {bad[0]}: {results[bad[0]]}")
    cyc = [results[i][1] for i in idx if results[i][1] > 0]
    if cyc:
        med = sorted(cyc)[len(cyc) // 2]
        print(f"median method.execute(): {med:,} cycles = {med / 168e3:.1f} ms at 168 MHz")
    if args.out:
        summary["results"] = [{"idx": i, "label": int(labels[i]), "pred": results[i][0],
                               "cycles": results[i][1], "status": results[i][2]} for i in idx]
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
