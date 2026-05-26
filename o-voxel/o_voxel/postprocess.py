from typing import *
from tqdm import tqdm
import numpy as np
import torch
import cv2
from PIL import Image
import trimesh
import trimesh.visual
import time
from flex_gemm.ops.grid_sample import grid_sample_3d
import nvdiffrast.torch as dr
import cumesh


_SCIPY_NDIMAGE = None
_SCIPY_NDIMAGE_CHECKED = False


def _get_scipy_ndimage():
    global _SCIPY_NDIMAGE, _SCIPY_NDIMAGE_CHECKED
    if not _SCIPY_NDIMAGE_CHECKED:
        try:
            from scipy import ndimage
            _SCIPY_NDIMAGE = ndimage
        except Exception:
            _SCIPY_NDIMAGE = None
        _SCIPY_NDIMAGE_CHECKED = True
    return _SCIPY_NDIMAGE


def _six_connected_structure() -> np.ndarray:
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, 1, 1] = True
    structure[0, 1, 1] = True
    structure[2, 1, 1] = True
    structure[1, 0, 1] = True
    structure[1, 2, 1] = True
    structure[1, 1, 0] = True
    structure[1, 1, 2] = True
    return structure


def _binary_dilate_6(matrix: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Dilate a 3D boolean volume using 6-connected neighbors."""
    matrix = matrix.astype(bool, copy=False)
    iterations = max(0, int(iterations))
    if iterations == 0:
        return matrix
    ndimage = _get_scipy_ndimage()
    if ndimage is not None:
        return ndimage.binary_dilation(
            matrix,
            structure=_six_connected_structure(),
            iterations=iterations,
        )
    for _ in range(iterations):
        padded = np.pad(matrix, 1, mode="constant", constant_values=False)
        matrix = (
            padded[1:-1, 1:-1, 1:-1] |
            padded[:-2, 1:-1, 1:-1] |
            padded[2:, 1:-1, 1:-1] |
            padded[1:-1, :-2, 1:-1] |
            padded[1:-1, 2:, 1:-1] |
            padded[1:-1, 1:-1, :-2] |
            padded[1:-1, 1:-1, 2:]
        )
    return matrix


def _fill_enclosed_voxels(occupied: np.ndarray) -> np.ndarray:
    """Fill every voxel that cannot be reached from the padded volume exterior."""
    occupied = occupied.astype(bool, copy=False)
    free = ~occupied
    seed = np.zeros_like(free, dtype=bool)
    seed[0, :, :] = free[0, :, :]
    seed[-1, :, :] = free[-1, :, :]
    seed[:, 0, :] = free[:, 0, :]
    seed[:, -1, :] = free[:, -1, :]
    seed[:, :, 0] = free[:, :, 0]
    seed[:, :, -1] = free[:, :, -1]

    ndimage = _get_scipy_ndimage()
    if ndimage is not None:
        exterior = ndimage.binary_propagation(
            seed,
            structure=_six_connected_structure(),
            mask=free,
        )
        return ~exterior

    exterior = seed.copy()
    frontier = seed
    while bool(frontier.any()):
        frontier = _binary_dilate_6(frontier, 1) & free & ~exterior
        exterior |= frontier

    return ~exterior


def _span_fill_along_axis(occupied: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(occupied.astype(bool, copy=False), axis, 0)
    any_occupied = moved.any(axis=0)
    first = np.argmax(moved, axis=0)
    last = moved.shape[0] - 1 - np.argmax(moved[::-1], axis=0)
    line = np.arange(moved.shape[0]).reshape((-1,) + (1,) * (moved.ndim - 1))
    filled = any_occupied.reshape((1,) + any_occupied.shape) & (line >= first) & (line <= last)
    return np.moveaxis(filled, 0, axis)


def _axis_span_fill_voxels(occupied: np.ndarray, min_axis_votes: int = 2) -> np.ndarray:
    """Fill voxels that lie between surface hits along multiple principal axes."""
    spans = [_span_fill_along_axis(occupied, axis) for axis in range(3)]
    return (spans[0].astype(np.uint8) + spans[1].astype(np.uint8) + spans[2].astype(np.uint8)) >= min_axis_votes


def _solid_fill_voxels(surface: np.ndarray, mode: str = "auto") -> np.ndarray:
    mode = (mode or "auto").lower()
    flood_filled = _fill_enclosed_voxels(surface)
    if mode == "flood":
        return flood_filled

    axis_filled = _axis_span_fill_voxels(surface, min_axis_votes=2)
    if mode in {"axis", "aggressive", "span"}:
        return flood_filled | axis_filled

    if mode == "auto":
        surface_count = max(1, int(surface.sum()))
        if int(flood_filled.sum()) < surface_count * 1.5:
            return flood_filled | axis_filled
        return flood_filled

    raise ValueError(f"Unsupported printable solid fill mode: {mode}")


def _boundary_mesh_from_voxels(
    voxels: np.ndarray,
    transform: np.ndarray,
    index_offset: int = 0,
) -> trimesh.Trimesh:
    """Convert filled voxels to a watertight boundary mesh without optional deps."""
    voxels = voxels.astype(bool, copy=False)
    padded = np.pad(voxels, 1, mode="constant", constant_values=False)
    core = padded[1:-1, 1:-1, 1:-1]
    quads = []

    def add_quads(coords: np.ndarray, axis: int, positive: bool) -> None:
        if coords.size == 0:
            return
        i, j, k = coords[:, 0], coords[:, 1], coords[:, 2]
        q = np.empty((coords.shape[0], 4, 3), dtype=np.int32)

        if axis == 0 and not positive:
            x = i
            q[:, 0] = np.stack([x, j, k], axis=1)
            q[:, 1] = np.stack([x, j, k + 1], axis=1)
            q[:, 2] = np.stack([x, j + 1, k + 1], axis=1)
            q[:, 3] = np.stack([x, j + 1, k], axis=1)
        elif axis == 0 and positive:
            x = i + 1
            q[:, 0] = np.stack([x, j, k], axis=1)
            q[:, 1] = np.stack([x, j + 1, k], axis=1)
            q[:, 2] = np.stack([x, j + 1, k + 1], axis=1)
            q[:, 3] = np.stack([x, j, k + 1], axis=1)
        elif axis == 1 and not positive:
            y = j
            q[:, 0] = np.stack([i, y, k], axis=1)
            q[:, 1] = np.stack([i + 1, y, k], axis=1)
            q[:, 2] = np.stack([i + 1, y, k + 1], axis=1)
            q[:, 3] = np.stack([i, y, k + 1], axis=1)
        elif axis == 1 and positive:
            y = j + 1
            q[:, 0] = np.stack([i, y, k], axis=1)
            q[:, 1] = np.stack([i, y, k + 1], axis=1)
            q[:, 2] = np.stack([i + 1, y, k + 1], axis=1)
            q[:, 3] = np.stack([i + 1, y, k], axis=1)
        elif axis == 2 and not positive:
            z = k
            q[:, 0] = np.stack([i, j, z], axis=1)
            q[:, 1] = np.stack([i, j + 1, z], axis=1)
            q[:, 2] = np.stack([i + 1, j + 1, z], axis=1)
            q[:, 3] = np.stack([i + 1, j, z], axis=1)
        else:
            z = k + 1
            q[:, 0] = np.stack([i, j, z], axis=1)
            q[:, 1] = np.stack([i + 1, j, z], axis=1)
            q[:, 2] = np.stack([i + 1, j + 1, z], axis=1)
            q[:, 3] = np.stack([i, j + 1, z], axis=1)

        quads.append(q)

    add_quads(np.argwhere(core & ~padded[:-2, 1:-1, 1:-1]), axis=0, positive=False)
    add_quads(np.argwhere(core & ~padded[2:, 1:-1, 1:-1]), axis=0, positive=True)
    add_quads(np.argwhere(core & ~padded[1:-1, :-2, 1:-1]), axis=1, positive=False)
    add_quads(np.argwhere(core & ~padded[1:-1, 2:, 1:-1]), axis=1, positive=True)
    add_quads(np.argwhere(core & ~padded[1:-1, 1:-1, :-2]), axis=2, positive=False)
    add_quads(np.argwhere(core & ~padded[1:-1, 1:-1, 2:]), axis=2, positive=True)

    if not quads:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)

    quads = np.concatenate(quads, axis=0)
    unique_corners, inverse = np.unique(quads.reshape(-1, 3), axis=0, return_inverse=True)
    quads = inverse.reshape(-1, 4)
    faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=0)

    corner_coords = unique_corners.astype(np.float64) + float(index_offset) - 0.5
    corners_h = np.concatenate([corner_coords, np.ones((corner_coords.shape[0], 1))], axis=1)
    vertices = (corners_h @ np.asarray(transform, dtype=np.float64).T)[:, :3]

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def clean_trimesh_for_export(mesh: trimesh.Trimesh, fix_normals: bool = False) -> trimesh.Trimesh:
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    elif hasattr(mesh, "unique_faces") and hasattr(mesh, "update_faces"):
        mesh.update_faces(mesh.unique_faces())

    if hasattr(mesh, "remove_unreferenced_vertices"):
        mesh.remove_unreferenced_vertices()

    if fix_normals:
        try:
            trimesh.repair.fix_normals(mesh)
        except Exception:
            if hasattr(mesh, "unify_face_orientations"):
                mesh.unify_face_orientations()
            if hasattr(mesh, "compute_vertex_normals"):
                mesh.compute_vertex_normals()
            else:
                _ = mesh.vertex_normals

    return mesh


def _project_vertices_to_source_mesh(
    vertices: np.ndarray,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    max_distance: Optional[float],
    verbose: bool = False,
) -> np.ndarray:
    if not torch.cuda.is_available():
        if verbose:
            print("Skipping printable projection: CUDA is not available")
        return vertices

    try:
        src_vertices = torch.as_tensor(source_vertices, dtype=torch.float32, device="cuda")
        src_faces = torch.as_tensor(source_faces, dtype=torch.int32, device="cuda")
        query = torch.as_tensor(vertices, dtype=torch.float32, device="cuda")
        bvh = cumesh.cuBVH(src_vertices, src_faces)
        projected_chunks = []

        for i in range(0, query.shape[0], 1000000):
            query_chunk = query[i:i + 1000000]
            distance, face_id, uvw = bvh.unsigned_distance(query_chunk, return_uvw=True)
            tri_vertices = src_vertices[src_faces[face_id.long()]]
            projected = (tri_vertices * uvw.unsqueeze(-1)).sum(dim=1)
            if max_distance is not None:
                keep_projection = distance <= max_distance
                projected = torch.where(keep_projection.unsqueeze(-1), projected, query_chunk)
            projected_chunks.append(projected.cpu())

        return torch.cat(projected_chunks, dim=0).numpy()
    except Exception as exc:
        if verbose:
            print(f"Skipping printable projection: {exc}")
        return vertices


def solidify_mesh_for_printing(
    vertices: Union[torch.Tensor, np.ndarray],
    faces: Union[torch.Tensor, np.ndarray],
    resolution: int = 256,
    shell_dilation: int = 1,
    max_voxels: int = 256 ** 3,
    project_back: bool = True,
    project_distance_voxels: float = 2.5,
    fill_mode: str = "auto",
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rebuild a mesh as a filled, watertight voxel solid for 3D-printable export.

    This is intentionally a post-process: it preserves the generated exterior
    where possible, but enclosed internal shells/cavities are treated as solid.
    """
    resolution = int(resolution)
    shell_dilation = int(shell_dilation)
    max_voxels = max(1, int(max_voxels))
    device = vertices.device if isinstance(vertices, torch.Tensor) else None
    vertices_np = vertices.detach().cpu().numpy() if isinstance(vertices, torch.Tensor) else np.asarray(vertices)
    faces_np = faces.detach().cpu().numpy() if isinstance(faces, torch.Tensor) else np.asarray(faces)

    source_mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
    max_extent = float(source_mesh.extents.max())
    if not np.isfinite(max_extent) or max_extent <= 0:
        raise ValueError("Cannot solidify a mesh with empty or degenerate bounds")

    pitch = max_extent / max(1, int(resolution))
    estimated_shape = np.ceil(np.maximum(source_mesh.extents, pitch) / pitch).astype(np.int64) + 2 * (shell_dilation + 2)
    estimated_voxels = int(np.prod(estimated_shape))
    if estimated_voxels > max_voxels:
        pitch *= (estimated_voxels / max_voxels) ** (1.0 / 3.0)
        if verbose:
            print(f"Printable solid voxel grid capped at {max_voxels} cells; using pitch {pitch:.6f}")

    voxel_grid = source_mesh.voxelized(pitch, method="subdivide")
    surface = voxel_grid.matrix.astype(bool, copy=False)
    pad_width = max(1, int(shell_dilation) + 2)
    surface = np.pad(surface, pad_width, mode="constant", constant_values=False)
    if shell_dilation > 0:
        surface = _binary_dilate_6(surface, shell_dilation)

    filled = _solid_fill_voxels(surface, mode=fill_mode)
    solid_mesh = _boundary_mesh_from_voxels(
        filled,
        voxel_grid.transform,
        index_offset=-pad_width,
    )

    if solid_mesh.faces.shape[0] == 0:
        raise ValueError("Solidification produced an empty mesh")

    clean_trimesh_for_export(solid_mesh)
    if project_back:
        max_project_distance = None
        if project_distance_voxels is not None and project_distance_voxels > 0:
            max_project_distance = pitch * float(project_distance_voxels)
        solid_mesh.vertices = _project_vertices_to_source_mesh(
            solid_mesh.vertices,
            vertices_np,
            faces_np,
            max_distance=max_project_distance,
            verbose=verbose,
        )
    clean_trimesh_for_export(solid_mesh, fix_normals=True)

    if verbose:
        print(
            "Printable solid mesh: "
            f"{solid_mesh.vertices.shape[0]} vertices, {solid_mesh.faces.shape[0]} faces"
        )

    out_vertices = torch.as_tensor(solid_mesh.vertices, dtype=torch.float32)
    out_faces = torch.as_tensor(solid_mesh.faces, dtype=torch.int32)
    if device is not None:
        out_vertices = out_vertices.to(device)
        out_faces = out_faces.to(device)
    return out_vertices, out_faces


def to_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    solidify: bool = False,
    solidify_resolution: int = 256,
    solidify_shell_dilation: int = 1,
    solidify_max_voxels: int = 256 ** 3,
    solidify_project_back: bool = True,
    solidify_project_distance_voxels: float = 2.5,
    solidify_fill_mode: str = "auto",
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose: bool = False,
    use_tqdm: bool = False,
):
    """
    Convert an extracted mesh to a GLB file.
    Performs cleaning, optional remeshing, UV unwrapping, and texture baking from a volume.
    
    Args:
        vertices: (N, 3) tensor of vertex positions
        faces: (M, 3) tensor of vertex indices
        attr_volume: (L, C) features of a sprase tensor for attribute interpolation
        coords: (L, 3) tensor of coordinates for each voxel
        attr_layout: dictionary of slice objects for each attribute
        aabb: (2, 3) tensor of minimum and maximum coordinates of the volume
        voxel_size: (3,) tensor of size of each voxel
        grid_size: (3,) tensor of number of voxels in each dimension
        decimation_target: target number of vertices for mesh simplification
        texture_size: size of the texture for baking
        remesh: whether to perform remeshing
        remesh_band: size of the remeshing band
        remesh_project: projection factor for remeshing
        solidify: whether to rebuild the mesh as a filled watertight solid for printing
        solidify_resolution: target voxel resolution along the longest mesh axis
        solidify_shell_dilation: shell dilation iterations used to seal small cracks before filling
        solidify_max_voxels: safety cap for CPU solidification grid size
        solidify_project_back: project the solid outer surface back to the source mesh for detail recovery
        solidify_project_distance_voxels: maximum projection distance measured in solidification voxels
        solidify_fill_mode: solid fill mode, one of "auto", "flood", or "aggressive"
        mesh_cluster_threshold_cone_half_angle_rad: threshold for cone-based clustering in uv unwrapping
        mesh_cluster_refine_iterations: number of iterations for refining clusters in uv unwrapping
        mesh_cluster_global_iterations: number of global iterations for clustering in uv unwrapping
        mesh_cluster_smooth_strength: strength of smoothing for clustering in uv unwrapping
        verbose: whether to print verbose messages
        use_tqdm: whether to use tqdm to display progress bar
    """
    # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
    # Timing helpers
    t0 = time.perf_counter()
    t_prev = t0
    def _log(stage: str):
        nonlocal t_prev
        now = time.perf_counter()
        print(f"[to_glb] {stage}: {now - t_prev:.3f}s (step), {now - t0:.3f}s (total)")
        t_prev = now

    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)
    assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
    assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
    assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
    assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

    # Calculate grid dimensions based on AABB and voxel size
    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size
    
    # Assertions for dimensions
    assert isinstance(voxel_size, torch.Tensor)
    assert voxel_size.dim() == 1 and voxel_size.size(0) == 3
    assert isinstance(grid_size, torch.Tensor)
    assert grid_size.dim() == 1 and grid_size.size(0) == 3
    _log("input_normalization")
    
    if use_tqdm:
        pbar = tqdm(total=6, desc="Extracting GLB")
    if verbose:
        print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # Start timing the heavy operations
    _log("starting")

    texture_source_vertices = vertices
    texture_source_faces = faces
    if solidify:
        vertices, faces = solidify_mesh_for_printing(
            vertices,
            faces,
            resolution=solidify_resolution,
            shell_dilation=solidify_shell_dilation,
            max_voxels=solidify_max_voxels,
            project_back=solidify_project_back,
            project_distance_voxels=solidify_project_distance_voxels,
            fill_mode=solidify_fill_mode,
            verbose=verbose,
        )
        _log("solidify_for_printing")

    # Move data to GPU
    vertices = vertices.cuda()
    faces = faces.cuda()
    
    # Initialize CUDA mesh handler
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    _log("mesh_init")
    
    # --- Initial Mesh Cleaning ---
    # Fills holes as much as we can before processing
    mesh.fill_holes(max_hole_perimeter=3e-2)
    if verbose:
        print(f"After filling holes: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
    vertices, faces = mesh.read()
    _log("initial_fill_and_read")
    if use_tqdm:
        pbar.update(1)
        
    # Build BVH for the current mesh to guide remeshing
    if use_tqdm:
        pbar.set_description("Building BVH")
    if verbose:
        print(f"Building BVH for current mesh...", end='', flush=True)
    bvh = cumesh.cuBVH(vertices, faces)
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    _log("bvh_build")

    if solidify:
        texture_vertices = texture_source_vertices.cuda()
        texture_faces = texture_source_faces.cuda()
        texture_mesh = cumesh.CuMesh()
        texture_mesh.init(texture_vertices, texture_faces)
        texture_mesh.fill_holes(max_hole_perimeter=3e-2)
        texture_vertices, texture_faces = texture_mesh.read()
        texture_bvh = cumesh.cuBVH(texture_vertices, texture_faces)
        _log("texture_bvh_build")
    else:
        texture_vertices = vertices
        texture_faces = faces
        texture_bvh = bvh
        
    if use_tqdm:
        pbar.set_description("Cleaning mesh")
    if verbose:
        print("Cleaning mesh...")
    
    # --- Branch 1: Standard Pipeline (Simplification & Cleaning) ---
    if not remesh:
        # Step 1: Aggressive simplification (3x target)
        mesh.simplify(decimation_target * 3, verbose=verbose)
        if verbose:
            print(f"After inital simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        
        # Step 2: Clean up topology (duplicates, non-manifolds, isolated parts)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"After initial cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
            
        # Step 3: Final simplification to target count
        mesh.simplify(decimation_target, verbose=verbose)
        if verbose:
            print(f"After final simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        
        # Step 4: Final Cleanup loop
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"After final cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
            
        # Step 5: Unify face orientations
        mesh.unify_face_orientations()
    
    # --- Branch 2: Remeshing Pipeline ---
    else:
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        resolution = grid_size.max().item()
        
        # Perform Dual Contouring remeshing (rebuilds topology)
        mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
            vertices, faces,
            center = center,
            scale = (resolution + 3 * remesh_band) / resolution * scale,
            resolution = resolution,
            band = remesh_band,
            project_back = remesh_project, # Snaps vertices back to original surface
            verbose = verbose,
            bvh = bvh,
        ))
        if verbose:
            print(f"After remeshing: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        _log("remesh_complete")
        
        # Simplify and clean the remeshed result (similar logic to above)
        mesh.simplify(decimation_target, verbose=verbose)
        if verbose:
            print(f"After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        _log("remesh_simplify")
    
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    _log("simplify_or_remesh_finished")
        
    
    # --- UV Parameterization ---
    if use_tqdm:
        pbar.set_description("Parameterizing new mesh")
    if verbose:
        print("Parameterizing new mesh...")
    
    out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
            "refine_iterations": mesh_cluster_refine_iterations,
            "global_iterations": mesh_cluster_global_iterations,
            "smooth_strength": mesh_cluster_smooth_strength,
        },
        return_vmaps=True,
        verbose=verbose,
    )
    out_vertices = out_vertices.cuda()
    out_faces = out_faces.cuda()
    out_uvs = out_uvs.cuda()
    out_vmaps = out_vmaps.cuda()
    mesh.compute_vertex_normals()
    out_normals = mesh.read_vertex_normals()[out_vmaps]
    _log("uv_unwrap_and_normals")
    
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    
    # --- Texture Baking (Attribute Sampling) ---
    if use_tqdm:
        pbar.set_description("Sampling attributes")
    if verbose:
        print("Sampling attributes...", end='', flush=True)
        
    # Setup differentiable rasterizer context
    ctx = dr.RasterizeCudaContext()
    # Prepare UV coordinates for rasterization (rendering in UV space)
    uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
    rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)
    
    # Rasterize in chunks to save memory
    for i in range(0, out_faces.shape[0], 100000):
        rast_chunk, _ = dr.rasterize(
            ctx, uvs_rast, out_faces[i:i+100000],
            resolution=[texture_size, texture_size],
        )
        mask_chunk = rast_chunk[..., 3:4] > 0
        rast_chunk[..., 3:4] += i # Store face ID in alpha channel
        rast = torch.where(mask_chunk, rast_chunk, rast)
    _log("rasterize_loop")
    
    # Mask of valid pixels in texture
    mask = rast[0, ..., 3] > 0
    
    # Interpolate 3D positions in UV space (finding 3D coord for every texel)
    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    valid_pos = pos[mask]
    
    # Map these positions back to the *original* high-res mesh to get accurate attributes
    # This corrects geometric errors introduced by simplification/remeshing
    _, face_id, uvw = texture_bvh.unsigned_distance(valid_pos, return_uvw=True)
    orig_tri_verts = texture_vertices[texture_faces[face_id.long()]] # (N_new, 3, 3)
    valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
    
    # Trilinear sampling from the attribute volume (Color, Material props)
    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
    attrs[mask] = grid_sample_3d(
        attr_volume,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
        grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
        mode='trilinear',
    )
    _log("attribute_sampling")
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    
    # --- Texture Post-Processing & Material Construction ---
    if use_tqdm:
        pbar.set_description("Finalizing mesh")
    if verbose:
        print("Finalizing mesh...", end='', flush=True)
    
    _log("start_finalizing")
    # Helpful debug prints (only when verbose)
    if verbose:
        try:
            print(f"attrs.shape={tuple(attrs.shape)}, texture_size={texture_size}")
            print(f"attr_layout keys={list(attr_layout.keys())}")
        except Exception:
            pass

    mask = mask.cpu().numpy()
    if verbose:
        print(f"mask pixels: {np.count_nonzero(mask)} / {mask.size}")
    _log("mask_converted")

    # Extract channels based on layout (BaseColor, Metallic, Roughness, Alpha)
    _log("extract_channels_start")
    base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha_mode = 'OPAQUE'
    _log("extract_channels_done")
    if verbose:
        print(f"channel shapes: base_color={base_color.shape}, metallic={metallic.shape}, roughness={roughness.shape}, alpha={alpha.shape}")

    # Inpainting: fill gaps (dilation) to prevent black seams at UV boundaries
    mask_inv = (~mask).astype(np.uint8)
    if verbose:
        print(f"mask_inv nonzero: {mask_inv.sum()}")

    _log("before_inpaint_base_color")
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    _log("after_inpaint_base_color")

    _log("before_inpaint_metallic")
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    _log("after_inpaint_metallic")

    _log("before_inpaint_roughness")
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    _log("after_inpaint_roughness")

    _log("before_inpaint_alpha")
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    _log("after_inpaint_alpha")
    
    # Create PBR material
    # Standard PBR packs Metallic and Roughness into Blue and Green channels
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        doubleSided=True if not remesh else False,
    )
    
    # --- Coordinate System Conversion & Final Object ---
    vertices_np = out_vertices.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    uvs_np = out_uvs.cpu().numpy()
    normals_np = out_normals.cpu().numpy()
    
    # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
    uvs_np[:, 1] = 1 - uvs_np[:, 1] # Flip UV V-coordinate
    
    textured_mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material)
    )
    _log("material_and_mesh_build")
    
    if use_tqdm:
        pbar.update(1)
        pbar.close()
    if verbose:
        print("Done")
    _log("complete")
    
    return textured_mesh
def remesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    solidify: bool = False,
    solidify_resolution: int = 256,
    solidify_shell_dilation: int = 1,
    solidify_max_voxels: int = 256 ** 3,
    solidify_project_back: bool = True,
    solidify_project_distance_voxels: float = 2.5,
    solidify_fill_mode: str = "auto",
    verbose: bool = False,
):
    """
    Standalone remeshing functionality detached from the full PBR rendering pipeline.
    """
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=vertices.device)
    assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
    assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
    assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
    assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

    # Calculate grid dimensions based on AABB and voxel size
    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=vertices.device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=vertices.device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size

    if solidify:
        vertices, faces = solidify_mesh_for_printing(
            vertices,
            faces,
            resolution=solidify_resolution,
            shell_dilation=solidify_shell_dilation,
            max_voxels=solidify_max_voxels,
            project_back=solidify_project_back,
            project_distance_voxels=solidify_project_distance_voxels,
            fill_mode=solidify_fill_mode,
            verbose=verbose,
        )

    vertices = vertices.cuda()
    faces = faces.cuda()
    
    # Initialize CUDA mesh handler
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    
    # --- Initial Mesh Cleaning ---
    mesh.fill_holes(max_hole_perimeter=3e-2)
    vertices, faces = mesh.read()

    bvh = cumesh.cuBVH(vertices, faces)

    center = aabb.mean(dim=0)
    scale = (aabb[1] - aabb[0]).max().item()
    resolution = grid_size.max().item()

    mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
        vertices, faces,
        center = center,
        scale = (resolution + 3 * remesh_band) / resolution * scale,
        resolution = resolution,
        band = remesh_band,
        project_back = remesh_project, # Snaps vertices back to original surface
        verbose = verbose,
        bvh = bvh,
    ))
    
    if verbose:
        print(f"After remeshing: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

    mesh.simplify(decimation_target, verbose=verbose)
    
    if verbose:
        print(f"After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

    out_vertices, out_faces = mesh.read()

    vertices_np = out_vertices.cpu().numpy()
    faces_np = out_faces.cpu().numpy()

    # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
    
    untextured_mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        process=False,
    )

    return untextured_mesh
