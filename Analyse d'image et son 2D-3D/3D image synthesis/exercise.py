
import numpy as np

import obj_io

def create_equilateral_tetrahedron(side_len):
    """Create a tetrahedron with all edges of the same length, ie each
    of its face must be an equilateral triangle.

    Returns
    -------

    vertices, triangles: tuple (array of 4 x 3 floats, array of 4 x 3 ints)
        The position of the vertices of the tetrahedron, and the triangles
        connecting the vertices.
    """
    vertices = np.zeros((4, 3))
    triangles = np.zeros((4, 3), dtype=np.uint32)

    # Exercise 1 code here: fill the vertices and triangles arrays
    # Activate backface culling in blender to make sure the faces are
    # oriented correctly. Beware of OBJ import options!
    # ...
    # end exercise code
    
    vertices[0,:] = np.array([1, 1, 1])
    vertices[1,:] = np.array([-1, -1, 1])
    vertices[2,:] = np.array([-1, 1, -1])
    vertices[3,:] = np.array([1, -1, -1])
    
    triangles[0,:] = np.array([0, 1, 2])
    triangles[1,:] = np.array([1, 2, 3])
    triangles[2,:] = np.array([0, 1, 3])
    triangles[3,:] = np.array([0, 2, 3])
    

    # locate origin on center of mass
    vertices -= vertices.mean(axis=0)
    return vertices, triangles

def compute_tri_normals(tri_points):
    """Compute the normals of the provided triangles.

    Parameters
    ----------

    tri_points: array of nb_triangles x 3 x 3 floats
        Location of the vertices for each triangle. The array dimensions are,
        respectively: triangle index, triangle vertex index, spatial dimension

    Returns
    -------

    tri_normals: array of nb_triangles x 3 floats
        Per-triangle normals.
    """
    # Start exercise code
    # ...
    # End exercise code
    raise NotImplementedError

def compute_normals(vertices, triangles):
    """Compute per-vertex normals for the provided triangle mesh.

    Parameters
    ----------

    vertices: array of nb_vertices x 3 floats
        The positions of the vertices of the mesh

    triangles: array of nb_triangles x 3 ints
        Triangles of the mesh
    """
    vert_normals = np.zeros((vertices.shape[0], 3))

    tri_normals = compute_tri_normals(vertices[triangles])

    # average the triangle normals
    # Start exercise code
    # ...
    # End exercise code

    vert_normals /= np.sqrt((vert_normals**2).sum(axis=1))[:, None]

    return vert_normals

def create_cylinder(
        radius, height, nb_vertical_subdiv=10, nb_angular_subdiv=20
    ):
    """Create a cylinder mesh.
    """

    nb_vertices = nb_angular_subdiv * nb_vertical_subdiv + 2
    vertices = np.zeros((nb_vertices, 3))
    vertices[-1] = [0., 0., height]

    nb_triangles = 2 * nb_angular_subdiv * nb_vertical_subdiv
    triangles = np.zeros((nb_triangles, 3), dtype=np.uint32)
    for angle_subdiv in range(nb_angular_subdiv):
        triangles[angle_subdiv] = [
            0,
            1 + (angle_subdiv + 1) % nb_angular_subdiv,
            1 + angle_subdiv,
        ]
        triangles[-angle_subdiv - 1] = [
            nb_vertices - 1,
            nb_vertices - ((angle_subdiv + 1) % nb_angular_subdiv) - 2,
            nb_vertices - angle_subdiv - 2,
        ]

    for vertical_subdiv in range(nb_vertical_subdiv):
        cur_height = float(vertical_subdiv) / (nb_vertical_subdiv-1.) * height
        for angle_subdiv in range(nb_angular_subdiv):
            theta = 2. * np.pi * float(angle_subdiv) / nb_angular_subdiv
            cur_vert = 1 + vertical_subdiv * nb_angular_subdiv + angle_subdiv
            vertices[cur_vert] = [
                radius * np.cos(theta), radius * np.sin(theta), cur_height,
            ]

            if vertical_subdiv + 1 >= nb_vertical_subdiv:
                continue
            next_vert = (
                1 + vertical_subdiv * nb_angular_subdiv
                + (angle_subdiv + 1) % nb_angular_subdiv
            )
            up_vert = 1 + (vertical_subdiv + 1) * nb_angular_subdiv + angle_subdiv
            upnext_vert = (
                1 + (vertical_subdiv + 1) * nb_angular_subdiv
                + (angle_subdiv + 1) % nb_angular_subdiv
            )
            triangles[
                nb_angular_subdiv * (2 * vertical_subdiv + 1)
                + 2 * angle_subdiv + 0
            ] = [cur_vert, next_vert, up_vert]
            triangles[
                nb_angular_subdiv * (2 * vertical_subdiv + 1)
                + 2 * angle_subdiv + 1
            ] = [next_vert, upnext_vert, up_vert]
    return vertices, triangles


def main():
    tetra_verts, tetra_tris = create_equilateral_tetrahedron(10.)
    obj_io.save_obj(r"d:\usuario\Daniel\Documentos\CS\tetra.obj", tetra_verts, tetra_tris)

    cylinder_verts, cylinder_tris = create_cylinder(3., 10.)
    try:
        cylinder_normals = compute_normals(cylinder_verts, cylinder_tris)
    except NotImplementedError:
        print("Warning, cylinder normals computation not implemented yet")
        cylinder_normals = None
    obj_io.save_obj(
        r"d:\usuario\Daniel\Documentos\CS\cylinder.obj", cylinder_verts, cylinder_tris, normals=cylinder_normals,
    )

if __name__ == "__main__":
    main()
