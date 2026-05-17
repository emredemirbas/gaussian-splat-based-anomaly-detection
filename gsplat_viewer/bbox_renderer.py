"""
OpenGL wireframe bounding box renderer.

Draws an axis-aligned bounding box (AABB) as 12 line segments
on top of the Gaussian splat rendering.
"""

from OpenGL import GL as gl
import numpy as np
import util


class BBoxRenderer:
    """Renders a wireframe AABB using GL_LINES."""

    # 12 edges of a cube, each edge is a pair of vertex indices (0–7)
    _EDGE_INDICES = np.array([
        0, 1,  1, 2,  2, 3,  3, 0,   # bottom face
        4, 5,  5, 6,  6, 7,  7, 4,   # top face
        0, 4,  1, 5,  2, 6,  3, 7,   # vertical edges
    ], dtype=np.uint32)

    def __init__(self):
        self.program = util.load_shaders(
            'shaders/bbox_vert.glsl',
            'shaders/bbox_frag.glsl',
        )
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        self.ebo = gl.glGenBuffers(1)
        self.active = False
        self.color = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # bright green

        # Upload index buffer (never changes)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER,
            self._EDGE_INDICES.nbytes,
            self._EDGE_INDICES,
            gl.GL_STATIC_DRAW,
        )
        gl.glBindVertexArray(0)

    def set_bbox(self, bbox_min: np.ndarray, bbox_max: np.ndarray):
        """
        Set the AABB to render.

        Parameters
        ----------
        bbox_min : (3,) float — minimum corner
        bbox_max : (3,) float — maximum corner
        """
        lo = np.asarray(bbox_min, dtype=np.float32)
        hi = np.asarray(bbox_max, dtype=np.float32)

        # 8 corners of the AABB
        vertices = np.array([
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ], dtype=np.float32)

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            vertices.nbytes,
            vertices,
            gl.GL_DYNAMIC_DRAW,
        )
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 0, None)
        gl.glEnableVertexAttribArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        self.active = True

    def clear(self):
        """Hide the bounding box."""
        self.active = False

    def draw(self, view_matrix: np.ndarray, projection_matrix: np.ndarray):
        """
        Draw the wireframe bounding box.

        Parameters
        ----------
        view_matrix : (4,4) float32 — from Camera.get_view_matrix()
        projection_matrix : (4,4) float32 — from Camera.get_project_matrix()
        """
        if not self.active:
            return

        gl.glUseProgram(self.program)

        # Set uniforms
        util.set_uniform_mat4(self.program, view_matrix, "view_matrix")
        util.set_uniform_mat4(self.program, projection_matrix, "projection_matrix")
        util.set_uniform_v3(self.program, self.color, "bbox_color")

        # Draw
        gl.glBindVertexArray(self.vao)
        gl.glLineWidth(3.0)
        gl.glDrawElements(
            gl.GL_LINES,
            len(self._EDGE_INDICES),
            gl.GL_UNSIGNED_INT,
            None,
        )
        gl.glBindVertexArray(0)
