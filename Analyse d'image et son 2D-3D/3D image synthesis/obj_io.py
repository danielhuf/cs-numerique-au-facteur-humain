
def save_obj(
        filename,
        vertices,
        triangles,
        normals=None,
        uvs=None,
        uv_triangles=None,
    ):
    """Save the given triangle mesh in the OBJ format.

    Parameters
    ----------

    vertices: array of nb_vertices x 3 floats
        The positions of the mesh vertices

    triangles: array of nb_triangles x 3 ints
        For each line, indices of the vertices forming a triangle in the mesh.

    normals: optional array of nb_vertices x 3 floats
        Per-vertex normal vector.

    uvs: optional array of nb_uv_vertices x 3 floats
        UV-space coordinates

    uv_triangles: array of nb_triangles x 3 ints
        Triangles in UV-space. Each UV triangle must correspond to a
        geometric triangle (but their indices will typically differ).
    """
    assert vertices.ndim == 2
    assert triangles.ndim == 2
    assert vertices.shape[1] == 3
    assert triangles.shape[1] == 3
    with open(filename, "w") as obj_file:
        for vertex in vertices:
            obj_file.write("v %f %f %f\n" % tuple(vertex))
        if normals is not None:
            assert normals.shape == vertices.shape
            for vertex_normal in normals:
                obj_file.write("vn %f %f %f\n" % tuple(vertex_normal))
        if uvs is not None:
            assert uvs.ndim == 2
            assert uvs.shape[1] == 2
            for uv in uvs:
                obj_file.write("vt %f %f %f\n" % tuple(uv))

        if uv_triangles is None:
            for triangle in triangles:
                obj_file.write("f %d %d %d\n" % tuple(triangle + 1))
        else:
            assert uv_triangles.shape == triangles.shape
            for triangle, uv_triangle in triangles:
                obj_file.write(
                    "f %d/%d %d/%d %d/%d\n"
                    % _interleave_tris(triangle + 1, uv_triangle + 1)
                )

def _interleave_tris(*args):
    return tuple(sum(map(list, zip(*args)), []))
