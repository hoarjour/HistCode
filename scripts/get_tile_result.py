import openslide
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py

if __name__ == "__main__":
    # read coord(.h5) file
    h5_file_path = "../tile_results/patches/tcga_lung/TCGA-63-7022-01A-01-BS1.b7eeb711-cd00-4172-ba09-a7f5c5eed805.h5"
    with h5py.File(h5_file_path, "r") as f:
        coords = f["coords"][()]

    # read wsi
    wsi_path = "../slides/TCGA-LUNG/TCGA-63-7022-01A-01-BS1.b7eeb711-cd00-4172-ba09-a7f5c5eed805.svs"
    wsi = openslide.open_slide(wsi_path)

    w, h = wsi.level_dimensions[0]
    level0_slide = wsi.read_region((0, 0), 0, (w, h))

    plt.imshow(level0_slide)
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()

    for idx, coord in enumerate(coords):
        rect = patches.Rectangle(coord, 256, 256, linewidth=0.3, edgecolor='black', fill=False)
        ax.add_patch(rect)

    plt.savefig("../tile_result/TCGA-LUNG/TCGA-63-7022-01A-01-BS1.b7eeb711-cd00-4172-ba09-a7f5c5eed805.png", dpi=500)
