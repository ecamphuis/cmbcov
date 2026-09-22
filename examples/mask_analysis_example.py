#!/usr/bin/env python3
"""
Example usage of the improved MaskWlm class.

This example demonstrates the key features of the reorganized MaskWlm class,
including mask loading, validation, power spectrum computation, and native
coupling matrix computation.
"""

import os
import tempfile

import healpy as hp
import numpy as np

from cmbcov.mask import MaskWlm


def create_example_mask(
    nside: int = 128, sky_fraction: float = 0.7, output_dir: str = ".", seed: int = 42
) -> str:
    """Create an example mask for demonstration."""
    npix = hp.nside2npix(nside)

    # Create a simple mask (randomly selected pixels)
    mask = np.zeros(npix, dtype=np.float32)
    n_good = int(npix * sky_fraction)
    np.random.seed(seed)
    n_start = np.random.randint(0, npix - n_good)
    mask[n_start : n_start + n_good] = 1.0

    # Add some gradual boundaries
    mask = hp.smoothing(mask, fwhm=np.radians(5.0))
    mask = np.clip(mask, 0.0, 1.0)

    # Save mask
    mask_file = "example_mask.fits"
    mask_path = os.path.join(output_dir, mask_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    hp.write_map(mask_path, mask, overwrite=True)

    return mask_path


def demonstrate_mask_analysis():
    """Demonstrate comprehensive mask analysis workflow."""

    print("=== MaskWlm Class Demonstration ===\n")

    # All outputs go to a scratch directory that is created up front, so that every
    # later step can rely on it existing regardless of which step failed.
    base_output_dir = os.path.join(tempfile.mkdtemp(prefix="mask_analysis_"), "output")
    output_dir = os.path.join(base_output_dir, "coupling_matrices")
    os.makedirs(output_dir, exist_ok=True)
    print(f"   Writing outputs to {base_output_dir}")

    # 1. Create and load mask
    print("1. Creating example mask...")
    mask_file = create_example_mask()

    try:
        # Initialize MaskWlm with new interface
        mask_analyzer = MaskWlm(
            mask_name=os.path.basename(mask_file),
            load_path=os.path.dirname(mask_file),
            precompute_alm=True,
        )

        print(f"   Loaded mask: {mask_analyzer}")

        # 2. Get mask statistics
        print("\n2. Mask validation and statistics...")

        try:
            mask_analyzer.validate_mask()
            print("   ✓ Mask validation passed")
        except ValueError as e:
            print(f"   ✗ Mask validation failed: {e}")

        stats = mask_analyzer.get_mask_statistics()
        print(f"   Sky fraction: {stats['sky_fraction']:.3f}")
        print(f"   Resolution: nside={stats['nside']}")
        print(f"   Good pixels: {stats['npix_good']}/{stats['npix_total']}")

        # 3. Compute power spectra
        print("\n3. Computing power spectra...")
        cl_mask, cl_mask_sq = mask_analyzer.compute_power_spectra(
            save=True,
            output_dir=base_output_dir,
        )
        print(f"   ✓ Computed power spectra (lmax={len(cl_mask)-1})")

        # 4. Degrade mask
        print("\n4. Mask degradation...")
        mask_analyzer.degrade_mask(target_nside=64)
        print("   ✓ Degraded mask to nside=64")

        # 5. Compute coupling matrices (native implementation)
        print("\n5. Computing coupling matrices...")
        try:
            # Master method kernels
            M, Msq = mask_analyzer.compute_master_coupling_kernels(
                l1max=120, l2max=120, output_dir=output_dir
            )
            print(f"   ✓ Computed master kernels: M{M.shape}, Msq{Msq.shape}")

            # PolSpice kernel
            K = mask_analyzer.compute_polspice_kernel(
                l1max=50, l2max=50, output_dir=output_dir
            )
            print(f"   ✓ Computed PolSpice kernel: K{K.shape}")

            # Combined PolSpice+Master kernel
            G = mask_analyzer.compute_polspice_master_kernel(
                l1max=50, l2max=50, output_dir=output_dir
            )
            print(f"   ✓ Computed combined kernel: G{G.shape}")

        except Exception as e:
            print(f"   ⚠ Error computing coupling matrices: {e}")

        # 6. Demonstrate high-level kernel access
        print("\n6. High-level kernel access...")
        try:
            # This will load existing kernels or compute if needed
            master_kernel = mask_analyzer.get_master_coupling_kernels(
                ell_max=50, kernel_dir=output_dir, use_squared=True
            )
            print(f"   ✓ Retrieved master kernel: {master_kernel.shape}")

        except Exception as e:
            print(f"   ⚠ Could not retrieve kernels: {e}")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_mask_analysis()
