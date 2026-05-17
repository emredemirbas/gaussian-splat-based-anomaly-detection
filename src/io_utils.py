import numpy as np
import struct
from math_utils import qvec2rotmat

def _read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)

def get_colmap_camera_centers(images_bin_path: str) -> np.ndarray:
    """
    Parse COLMAP images.bin and return camera centers (world coordinates), sorted by image name.

    COLMAP stores world-to-camera transform with rotation R and translation t.
    Camera center in world coords: C = -R^T t

    Returns
    -------
    centers : (N,3) float64
    """
    centers = []
    names = []

    with open(images_bin_path, "rb") as fid:
        num_reg_images = _read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_reg_images):
            _image_id = _read_next_bytes(fid, 4, "I")[0]
            qvec = np.array(_read_next_bytes(fid, 32, "dddd"), dtype=np.float64)
            tvec = np.array(_read_next_bytes(fid, 24, "ddd"), dtype=np.float64)
            _camera_id = _read_next_bytes(fid, 4, "I")[0]

            name_bytes = bytearray()
            while True:
                c = fid.read(1)
                if c == b"\x00" or c == b"":
                    break
                name_bytes.extend(c)
            image_name = name_bytes.decode("utf-8")

            num_points2d = _read_next_bytes(fid, 8, "Q")[0]
            fid.seek(num_points2d * 24, 1)

            R = qvec2rotmat(qvec)
            C = -R.T @ tvec

            centers.append(C)
            names.append(image_name)

    order = np.argsort(names)
    return np.array([centers[i] for i in order], dtype=np.float64)


def get_colmap_camera_poses(images_bin_path: str) -> dict:
    """
    Parse COLMAP images.bin and return full camera poses (world coordinates),
    sorted by image name.

    COLMAP stores world-to-camera transform with rotation R and translation t.
    Camera center in world coords: C = -R^T t

    Returns
    -------
    dict with keys:
        centers    : (N,3) float64  — camera centre positions in world frame
        rotations  : (N,3,3) float64 — world-to-camera rotation matrices R
        translations : (N,3) float64 — world-to-camera translation vectors t
        names      : list of str     — image filenames (sorted)
    """
    centers = []
    rotations = []
    translations = []
    names = []

    with open(images_bin_path, "rb") as fid:
        num_reg_images = _read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_reg_images):
            _image_id = _read_next_bytes(fid, 4, "I")[0]
            qvec = np.array(_read_next_bytes(fid, 32, "dddd"), dtype=np.float64)
            tvec = np.array(_read_next_bytes(fid, 24, "ddd"), dtype=np.float64)
            _camera_id = _read_next_bytes(fid, 4, "I")[0]

            name_bytes = bytearray()
            while True:
                c = fid.read(1)
                if c == b"\x00" or c == b"":
                    break
                name_bytes.extend(c)
            image_name = name_bytes.decode("utf-8")

            num_points2d = _read_next_bytes(fid, 8, "Q")[0]
            fid.seek(num_points2d * 24, 1)

            R = qvec2rotmat(qvec)
            C = -R.T @ tvec

            centers.append(C)
            rotations.append(R)
            translations.append(tvec)
            names.append(image_name)

    order = np.argsort(names)
    return {
        "centers": np.array([centers[i] for i in order], dtype=np.float64),
        "rotations": np.array([rotations[i] for i in order], dtype=np.float64),
        "translations": np.array([translations[i] for i in order], dtype=np.float64),
        "names": [names[i] for i in order],
    }


class PlyIO:
    """
    Minimal PLY reader/writer for vertex-only PLYs (ASCII or binary little/big endian).
    Stores vertices in a numpy structured array with fields matching the PLY properties.
    """

    @staticmethod
    def read(path: str) -> np.ndarray:
        with open(path, "rb") as f:
            header = []
            while True:
                line = f.readline()
                if not line:
                    raise ValueError("Unexpected EOF while reading PLY header")
                header.append(line.decode("utf-8", errors="replace").strip())
                if header[-1] == "end_header":
                    break

            if not header[0].startswith("ply"):
                raise ValueError("Not a PLY file")

            fmt = None
            vertex_count = None
            props = []
            in_vertex = False

            for line in header:
                if line.startswith("format "):
                    fmt = line.split()[1]
                elif line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                    in_vertex = True
                elif line.startswith("element ") and not line.startswith("element vertex"):
                    in_vertex = False
                elif in_vertex and line.startswith("property "):
                    parts = line.split()
                    if len(parts) >= 3:
                        props.append((parts[1], parts[2]))

            if fmt is None or vertex_count is None or vertex_count <= 0:
                raise ValueError("PLY header missing format or vertex count")

            ply2np = {
                "char": "i1", "uchar": "u1",
                "short": "i2", "ushort": "u2",
                "int": "i4", "uint": "u4",
                "float": "f4", "double": "f8",
            }

            dtype = []
            for t, name in props:
                if t not in ply2np:
                    raise ValueError(f"Unsupported PLY property type: {t}")
                dtype.append((name, np.dtype(ply2np[t])))

            if fmt == "ascii":
                data_txt = f.read().decode("utf-8", errors="replace").strip().splitlines()
                if len(data_txt) < vertex_count:
                    raise ValueError("Not enough vertex lines in ASCII PLY")
                rows = [line.strip().split() for line in data_txt[:vertex_count]]
                arr = np.zeros(vertex_count, dtype=dtype)
                for i, row in enumerate(rows):
                    for (t, name), val in zip(props, row):
                        arr[name][i] = np.array(val, dtype=arr.dtype[name])
                return arr

            if fmt in ("binary_little_endian", "binary_big_endian"):
                endian = "<" if fmt == "binary_little_endian" else ">"
                np_dtype = np.dtype([(n, np.dtype(dt).newbyteorder(endian)) for n, dt in dtype])
                return np.fromfile(f, dtype=np_dtype, count=vertex_count)

            raise ValueError(f"Unsupported PLY format: {fmt}")

    @staticmethod
    def save(path: str, data: np.ndarray, as_text: bool = False):
        if data.ndim != 1:
            raise ValueError("PLY data must be a 1D structured array of vertices")

        lines = ["ply"]
        lines.append("format ascii 1.0" if as_text else "format binary_little_endian 1.0")
        lines.append(f"element vertex {len(data)}")

        np2ply = {
            "i1": "char", "u1": "uchar",
            "i2": "short", "u2": "ushort",
            "i4": "int", "u4": "uint",
            "f4": "float", "f8": "double",
        }

        for name in data.dtype.names:
            kind = data.dtype[name].str[1:]  # '<f4' -> 'f4'
            if kind not in np2ply:
                raise ValueError(f"Unsupported dtype for PLY save: {data.dtype[name]}")
            lines.append(f"property {np2ply[kind]} {name}")

        lines.append("end_header")
        header = "\n".join(lines) + "\n"

        if as_text:
            with open(path, "w", encoding="utf-8") as f:
                f.write(header)
                names = data.dtype.names
                for i in range(len(data)):
                    f.write(" ".join(str(data[name][i]) for name in names) + "\n")
        else:
            with open(path, "wb") as f:
                f.write(header.encode("utf-8"))
                data.tofile(f)
