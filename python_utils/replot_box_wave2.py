import numpy as np
import xarray as xr
import argparse
import os
import matplotlib.pyplot as plt

def read_bin_component(file_path, shape):
    """Read a binary file and return reshaped float32 array."""
    data = np.fromfile(file_path, dtype=np.float32)
    if data.size != np.prod(shape):
        raise ValueError(f"Expected shape {shape}, but got {data.size} values.")
    return data.reshape(shape)

def main(args):
    # File naming convention
    u_path = os.path.join(args.bin_dir, f"{args.tag}_u.bin")
    v_path = os.path.join(args.bin_dir, f"{args.tag}_v.bin")
    w_path = os.path.join(args.bin_dir, f"{args.tag}_w.bin")

    print(f"Reading: {u_path}, {v_path}, {w_path}")
    shape = tuple(map(int, args.shape))  # (nx, ny, nz)

    # Read binary files
    u = read_bin_component(u_path, shape)
    v = read_bin_component(v_path, shape)
    w = read_bin_component(w_path, shape)

    nx, ny, nz = shape
    uvw_stack = np.stack([u, v, w], axis=0)  # Shape: (3, nx, ny, nz)

    da = xr.DataArray(
        uvw_stack,
        dims=("uvw", "x", "y", "z"),
        coords={
            "uvw": ["u", "v", "w"],
            "x": np.arange(nx) * args.dx,
            "y": np.arange(ny) * args.dy,
            "z": np.arange(nz) * args.dz
        }
    )
    da.attrs["dx"] = args.dx
    da.attrs["dy"] = args.dy
    da.attrs["dz"] = args.dz
    da.attrs["tag"] = args.tag

    # Make dir for plots
    outdir = "./plots/"
    os.makedirs(outdir, exist_ok=True)

    y_vals = da.coords['y'].values
    z_vals = da.coords['z'].values
    mid_y_index = ny // 2
    mid_y_val = y_vals[mid_y_index]

    # Plot of windfield
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 4)  # Create 4-column grid for flexible width ratio

    # Left subplot: Windfield (occupies 3/4 of figure width)
    ax0 = fig.add_subplot(gs[0, :3])
    da.sel(uvw='u', y=mid_y_val).plot(x='x', ax=ax0)
    ax0.set_title("Windfield at y = {:.1f} m".format(mid_y_val))
    ax0.grid(True)

    # Right subplot: Vertical wind profile over x at y=90
    ax1 = fig.add_subplot(gs[0, 3])
    u_slice = da.sel(uvw='u').isel(y=mid_y_index)  # Shape: (x, z)

    u_mean = u_slice.mean(dim='x')
    u_std = u_slice.std(dim='x')

    ax1.errorbar(u_mean, z_vals, xerr=u_std, fmt='o', color='cornflowerblue', ecolor='cornflowerblue', capsize=3)
    ax1.set_xlabel('U [m/s]')
    ax1.set_ylabel('Height z [m]')
    ax1.set_title("Profile at y = {:.1f} m".format(mid_y_val))
    ax1.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"R_Windfield_{args.tag}.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Replotting for {args.tag} done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct u,v,w field from HiperSim bin files")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--bin_dir", type=str, required=True)
    parser.add_argument("--shape", nargs=3, required=True, help="Shape as nx ny nz")
    parser.add_argument("--dx", type=float, required=True)
    parser.add_argument("--dy", type=float, default=6.0)
    parser.add_argument("--dz", type=float, default=6.0)
    args = parser.parse_args()

    main(args)
