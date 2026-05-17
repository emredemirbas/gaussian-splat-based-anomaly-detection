"""Write a COLMAP `images.bin` consumable by `io_utils.get_colmap_camera_centers`.

Binary layout (little-endian) per image:
    image_id     : <I  (4 B)
    qvec         : <dddd (32 B, order qw qx qy qz)
    tvec         : <ddd  (24 B)
    camera_id    : <I  (4 B)
    name + \\x00 : utf-8 null-terminated
    num_points2D : <Q  (8 B) — we always write 0
    (no points2D records)
"""

from __future__ import annotations

import struct
import numpy as np


def write_images_bin(
    path: str,
    qvecs: np.ndarray,        # (N, 4) qw qx qy qz
    tvecs: np.ndarray,        # (N, 3)
    names: list[str],
    camera_id: int = 1,
) -> None:
    n = len(names)
    assert qvecs.shape == (n, 4)
    assert tvecs.shape == (n, 3)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", n))
        for i in range(n):
            image_id = i + 1
            f.write(struct.pack("<I", image_id))
            f.write(struct.pack("<dddd",
                                float(qvecs[i, 0]),
                                float(qvecs[i, 1]),
                                float(qvecs[i, 2]),
                                float(qvecs[i, 3])))
            f.write(struct.pack("<ddd",
                                float(tvecs[i, 0]),
                                float(tvecs[i, 1]),
                                float(tvecs[i, 2])))
            f.write(struct.pack("<I", int(camera_id)))
            f.write(names[i].encode("utf-8"))
            f.write(b"\x00")
            f.write(struct.pack("<Q", 0))
