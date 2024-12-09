<p align="center">

  <img src="icon.png" width="200"/> 

  <h1 align="center">WGEO Converter</h1>

  <p align="center">
    A simple script that converts League of Legends WGEO files to GLB format
  </p>
</p>

### What it does

This script takes a WGEO file (League of Legends World Geometry) and converts it to GLB (glTF Binary), making it easy to view and edit in modern 3D software.

### How to use
Make sure you have Python installed.
Install required libraries:

```pip install numpy pygltflib```

### Run the script:

```py WGEO.py room.wgeo```

*Note: Ensure room.mat and a Textures folder are in the same directory as your WGEO file.*

### That's it!

The script will create a GLB file that you can open in Blender, Unity, or any other software that supports glTF.
