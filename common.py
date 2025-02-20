import json
import os.path
import numpy as np
import pylab as plt

def loadFromJSON(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = np.array(value)
    
    return data

def make_plots(prefix):
    # Create the output folder if it doesn't exist
    output_folder = "outputFig"
    os.makedirs(output_folder, exist_ok=True)

    # Get all open figure numbers
    figures = [plt.figure(i) for i in plt.get_fignums()]

    # Save each figure as a PDF
    for i, fig in enumerate(figures, start=1):
        fig.savefig(os.path.join(output_folder, f"{prefix}_figure_{i}.pdf"), format="pdf")

    print(f"Saved {len(figures)} figures to '{output_folder}'")
    return

def make_visual(title):
    # Create the output folder if it doesn't exist
    output_folder = "outputFig"
    os.makedirs(output_folder, exist_ok=True)

    # Get all open figure numbers
    figures = [plt.figure(i) for i in plt.get_fignums()]

    # Save each figure as a PDF
    for i, fig in enumerate(figures, start=1):
        fig.savefig(os.path.join(output_folder, f"visual_{title}.pdf"), format="pdf")

    print(f"Saved {len(figures)} figures to '{output_folder}'")
    return