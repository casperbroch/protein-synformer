# Functions for visualizing 3D structure
# Source: https://github.com/facebookresearch/esm  
import py3Dmol
from matplotlib import colormaps
import torch
from esm.sdk.api import ESMProtein

def visualize_pdb(pdb_string, style="cartoon"):
    view = py3Dmol.view(width=600, height=600)
    view.addModel(pdb_string, "pdb")
    view.setStyle({style: {"color": "spectrum"}})
    view.zoomTo()
    view.render()
    view.center()
    return view

def visualize_3D_coordinates(coordinates):
    """
    This uses all Alanines
    """
    protein_with_same_coords = ESMProtein(coordinates=coordinates)
    # pdb with all alanines
    pdb_string = protein_with_same_coords.to_pdb_string()
    return visualize_pdb(pdb_string)

def visualize_3D_protein(protein: ESMProtein, style="cartoon", savefig=None):
    pdb_string = protein.to_pdb_string()
    view = visualize_pdb(pdb_string, style=style)
    return view

# Code based on 3D SASA visualization, but adapted to highlight binding sites instead 
def visualize_binding_sites_3D_protein(protein: ESMProtein, style="cartoon", surface=False, cmap=colormaps["cividis"]):
    pdb_string = protein.to_pdb_string()
    view = py3Dmol.view(width=600, height=600)
    view.addModel(pdb_string, "pdb")
    if surface:
        # TODO: Different color for different regions of the surface?
        view.addSurface(
            0, # py3Dmol.SES, 
            {"opacity": 0.5, "color": "lightblue"}, 
            {"chain": "A"}
        )
    # Residue locations that are part of binding site are 1, others are 0 
    binding_sites_indicator = torch.zeros(len(protein.sequence))
    for site in binding_sites:
        binding_sites_indicator[site.start-1:site.end] = 1
    for res_pos, res_color in enumerate(get_color_strings(binding_sites_indicator, cmap)):
        view.setStyle(
            {"chain": "A", "resi": res_pos+1}, 
            {style: {"color": res_color}}
        )
    view.zoomTo()
    view.render()
    view.center()
    return view

def get_color_strings(binding_sites_indicator, cmap):
    rgbas = (cmap(binding_sites_indicator) * 255).astype(int)
    return [
        f"rgb({rgba[0]},{rgba[1]},{rgba[2]})" 
        for rgba in rgbas
    ]
