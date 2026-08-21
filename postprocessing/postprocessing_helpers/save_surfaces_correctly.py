########################################################################
###### IRINA GRIGORESCU
######
###### This is a helper function to read in and save a surface correctly for wb_view
######
########################################################################

import argparse
import nibabel as nib
from src.io import save_gifti_surface

def save_gifti_surface_correctly(data_path_in, data_path_out, surface_hemi, surface_type):
    # ------ print information ------
    print(f"\nRunning save_surface_correctly.py with:\n")
    print(f"INPUT       : {data_path_in}")
    print(f"OUTPUT      : {data_path_out}")
    print(f"INFO surface: {surface_hemi} | {surface_type}\n")

    v_in, f_in = nib.load(data_path_in).agg_data('pointset'), nib.load(data_path_in).agg_data('triangle')

    save_gifti_surface(v_in, f_in,
                       save_dir=data_path_out,
                       surf_hemi=surface_hemi, surf_type=surface_type)

def main():
    parser = argparse.ArgumentParser(description="Save Surface Correctly")

    parser.add_argument('--data_path_in', default=None, type=str, help="path to image to load")
    parser.add_argument('--data_path_out', default=None, type=str, help="path to image to save")
    parser.add_argument('--surface_hemi', default=None, type=str, help="[left right]")
    parser.add_argument('--surface_type', default=None, type=str, help="[wm pial midthickness inflated vinflated sphere]")

    args = parser.parse_args()

    save_gifti_surface_correctly(
        data_path_in=args.data_path_in,
        data_path_out=args.data_path_out,
        surface_hemi=args.surface_hemi,
        surface_type=args.surface_type
    )

    print("Done\n")

if __name__ == "__main__":
    main()