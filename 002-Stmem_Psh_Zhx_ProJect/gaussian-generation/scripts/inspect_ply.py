import argparse
import os
import sys
from plyfile import PlyData

def inspect_ply(path):
    if not os.path.exists(path):
        print(f"Error: File '{path}' not found.")
        return

    print(f"Inspecting PLY file: {path}")
    try:
        plydata = PlyData.read(path)
    except Exception as e:
        print(f"Error reading PLY file: {e}")
        return
    
    # Usually 3DGS stores data in the 'vertex' element
    vertex_element = None
    for element in plydata.elements:
        if element.name == 'vertex':
            vertex_element = element
            break
    
    if vertex_element:
        num_gaussians = vertex_element.count
        properties = vertex_element.properties
        property_names = [prop.name for prop in properties]
        
        print("-" * 50)
        print(f"Number of Gaussian Spheres (Vertices): {num_gaussians}")
        print("-" * 50)
        print(f"Total Number of Parameters (Properties): {len(property_names)}")
        print("-" * 50)
        print("Parameter Names:")
        for name in property_names:
            print(f"  - {name}")
        print("-" * 50)
    else:
        print("No 'vertex' element found in the PLY file.")
        print("Elements found:", [e.name for e in plydata.elements])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a 3DGS PLY file to show Gaussian count and parameters.")
    parser.add_argument("path", type=str, help="Path to the .ply file")
    
    args = parser.parse_args()
    inspect_ply(args.path)
