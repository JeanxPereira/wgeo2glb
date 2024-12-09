import struct
from dataclasses import dataclass
from typing import List, Dict
import io
import os
import sys
import argparse
import numpy as np
from pygltflib import *

class WGEOReadException(Exception):
    """Custom exception for WGEO file reading errors"""
    pass

@dataclass
class MaterialInfo:
    name: str
    texture: str
    opacity: int
    color: List[int]
    disable_backface_culling: bool

class MaterialReader:
    @staticmethod
    def read_mat_file(filepath: str) -> Dict[str, MaterialInfo]:
        materials = {}
        current_material = None

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('[MaterialBegin]'):
                    current_material = {}
                elif line.startswith('[MaterialEnd]'):
                    if 'Name' in current_material:
                        name = current_material['Name']
                        texture = current_material.get('Texture', '')
                        opacity = int(current_material.get('Opacity', 255))
                        color = [int(x) for x in current_material.get('Color24', '204 204 204').split()]
                        disable_backface_culling = current_material.get('DisableBackfaceCulling', '0') == '1'
                        materials[name] = MaterialInfo(name, texture, opacity, color, disable_backface_culling)
                    current_material = None
                elif '=' in line and current_material is not None:
                    key, value = line.split('=', 1)
                    current_material[key.strip()] = value.strip()

        return materials
    
@dataclass
class WGEOVert:
    pos: List[float]
    uv: List[float]

@dataclass
class WGEOSubMesh:
    vertex_id: int
    vertex_offset: int
    vertex_count: int
    index_id: int
    index_offset: int
    index_count: int

@dataclass
class WGEOMesh:
    texture: str
    material: str
    sphere: List[float]
    min_bounds: List[float]
    max_bounds: List[float]
    flag0: int
    submeshes: List[WGEOSubMesh]
    verts: List[WGEOVert]
    inds: List[int]

@dataclass
class WGEOFile:
    FILE_MAGIC = b'WGEO'
    magic: bytes
    version: int
    model_count: int
    face_count: int
    meshes: List[WGEOMesh]

class WGEOReader:
    @staticmethod
    def read_string(stream: io.BytesIO, length: int) -> str:
        data = stream.read(length)
        return data.split(b'\x00')[0].decode('utf-8', errors='replace')

    @staticmethod
    def read_float(stream: io.BytesIO) -> float:
        return struct.unpack('<f', stream.read(4))[0]

    @staticmethod
    def read_int(stream: io.BytesIO) -> int:
        return struct.unpack('<i', stream.read(4))[0]

    @staticmethod
    def read_uint(stream: io.BytesIO) -> int:
        return struct.unpack('<I', stream.read(4))[0]

    @staticmethod
    def read_short(stream: io.BytesIO) -> int:
        return struct.unpack('<H', stream.read(2))[0]

    @classmethod
    def read_submesh(cls, stream: io.BytesIO) -> WGEOSubMesh:
        return WGEOSubMesh(
            vertex_id=cls.read_int(stream),
            vertex_offset=cls.read_int(stream),
            vertex_count=cls.read_int(stream),
            index_id=cls.read_int(stream),
            index_offset=cls.read_int(stream),
            index_count=cls.read_int(stream)
        )

    @classmethod
    def read_mesh(cls, stream: io.BytesIO) -> WGEOMesh:
        texture = cls.read_string(stream, 260)
        material = cls.read_string(stream, 64)
        sphere = [cls.read_float(stream) for _ in range(4)]
        min_bounds = [cls.read_float(stream) for _ in range(3)]
        max_bounds = [cls.read_float(stream) for _ in range(3)]
        
        vertex_count = cls.read_int(stream)
        index_count = cls.read_int(stream)
        
        if not 0 <= vertex_count <= 1000000 or not 0 <= index_count <= 1000000:
            raise WGEOReadException(f"Invalid vertex or index count: {vertex_count}, {index_count}")
        
        # Create vertex buffer memory
        vertex_size = 5 * 4  # 3 floats for position, 2 floats for UV
        vertex_buffer = stream.read(vertex_count * vertex_size)
        
        # Create index buffer memory
        index_format_size = 2  # Assuming 16-bit indices
        index_buffer = stream.read(index_count * index_format_size)
        
        # Parse vertices
        verts = []
        for i in range(vertex_count):
            offset = i * vertex_size
            pos = struct.unpack('<fff', vertex_buffer[offset:offset+12])
            uv = struct.unpack('<ff', vertex_buffer[offset+12:offset+20])
            verts.append(WGEOVert(pos=list(pos), uv=list(uv)))
        
        # Parse indices
        inds = struct.unpack(f'<{index_count}H', index_buffer)
        
        return WGEOMesh(
            texture=texture,
            material=material,
            sphere=sphere,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            flag0=0,  # We're not reading this value, so set it to 0
            submeshes=[],  # We're not reading submeshes, so leave it empty
            verts=verts,
            inds=list(inds)
        )

    @classmethod
    def read_file(cls, filepath: str) -> WGEOFile:
        with open(filepath, 'rb') as f:
            stream = io.BytesIO(f.read())
            
            magic = stream.read(4)
            if magic != WGEOFile.FILE_MAGIC:
                raise WGEOReadException(f"Wrong magic number: expected {WGEOFile.FILE_MAGIC}, got {magic}")
            
            version = cls.read_uint(stream)
            if version not in [4, 5]:
                raise WGEOReadException(f"Unsupported version: {version}")
            
            model_count = cls.read_int(stream)
            face_count = cls.read_uint(stream)
            
            meshes = []
            for _ in range(model_count):
                try:
                    mesh = cls.read_mesh(stream)
                    meshes.append(mesh)
                except Exception as e:
                    print(f"Error reading mesh: {str(e)}")
                    # You might want to break here if one mesh fails
                    # break
            
            return WGEOFile(magic=magic, version=version, model_count=model_count, face_count=face_count, meshes=meshes)

class GLTFExporter:
    @staticmethod
    def export(wgeo: WGEOFile, materials: Dict[str, MaterialInfo], output_file: str) -> GLTF2:
        gltf = GLTF2()
        gltf.asset = Asset(version="2.0", generator="WGEO Converter")
        gltf.scene = 0
        gltf.scenes = [Scene(nodes=list(range(len(wgeo.meshes))))]
        
        buffer_data = bytearray()
        material_indices = {}
        texture_indices = {}
        
        for mesh_idx, wgeo_mesh in enumerate(wgeo.meshes):
            vertices = [coord for vert in wgeo_mesh.verts for coord in vert.pos]
            uvs = [coord for vert in wgeo_mesh.verts for coord in vert.uv]
            indices = wgeo_mesh.inds
            
            vertex_bytes = np.array(vertices, dtype=np.float32).tobytes()
            uv_bytes = np.array(uvs, dtype=np.float32).tobytes()
            index_bytes = np.array(indices, dtype=np.uint16).tobytes()
            
            vertex_buffer_view = BufferView(
                buffer=0,
                byteOffset=len(buffer_data),
                byteLength=len(vertex_bytes),
                target=ARRAY_BUFFER
            )
            gltf.bufferViews.append(vertex_buffer_view)
            buffer_data.extend(vertex_bytes)
            
            uv_buffer_view = BufferView(
                buffer=0,
                byteOffset=len(buffer_data),
                byteLength=len(uv_bytes),
                target=ARRAY_BUFFER
            )
            gltf.bufferViews.append(uv_buffer_view)
            buffer_data.extend(uv_bytes)
            
            index_buffer_view = BufferView(
                buffer=0,
                byteOffset=len(buffer_data),
                byteLength=len(index_bytes),
                target=ELEMENT_ARRAY_BUFFER
            )
            gltf.bufferViews.append(index_buffer_view)
            buffer_data.extend(index_bytes)
            
            vertex_accessor = Accessor(
                bufferView=len(gltf.bufferViews) - 3,
                componentType=FLOAT,
                count=len(vertices) // 3,
                type="VEC3",
                min=[min(vertices[i::3]) for i in range(3)],
                max=[max(vertices[i::3]) for i in range(3)]
            )
            
            uv_accessor = Accessor(
                bufferView=len(gltf.bufferViews) - 2,
                componentType=FLOAT,
                count=len(uvs) // 2,
                type="VEC2",
                min=[min(uvs[i::2]) for i in range(2)],
                max=[max(uvs[i::2]) for i in range(2)]
            )
            
            index_accessor = Accessor(
                bufferView=len(gltf.bufferViews) - 1,
                componentType=UNSIGNED_SHORT,
                count=len(indices),
                type="SCALAR",
                min=[min(indices)],
                max=[max(indices)]
            )
            
            gltf.accessors.extend([vertex_accessor, uv_accessor, index_accessor])
            
            # Create material if it doesn't exist
            if wgeo_mesh.material not in material_indices:
                material_index = len(gltf.materials)
                mat_info = materials.get(wgeo_mesh.material, MaterialInfo(wgeo_mesh.material, '', 255, [204, 204, 204], False))
                
                material = Material()
                material.name = mat_info.name
                material.doubleSided = mat_info.disable_backface_culling
                material.alphaMode = "OPAQUE" if mat_info.opacity == 255 else "BLEND"
                
                pbr = PbrMetallicRoughness()
                pbr.baseColorFactor = [c/255 for c in mat_info.color] + [mat_info.opacity/255]
                pbr.metallicFactor = 0.0
                pbr.roughnessFactor = 1.0
                
                material.pbrMetallicRoughness = pbr
                
                if mat_info.texture:
                    if mat_info.texture not in texture_indices:
                        texture_index = len(gltf.textures)
                        image_index = len(gltf.images)
                        
                        texture = Texture(source=image_index)
                        gltf.textures.append(texture)
                        
                        image = Image(
                            uri=f"Textures/{mat_info.texture}",
                            name=mat_info.texture
                        )
                        gltf.images.append(image)
                        
                        texture_indices[mat_info.texture] = texture_index
                    
                    material.pbrMetallicRoughness.baseColorTexture = TextureInfo(index=texture_indices[mat_info.texture])
                
                gltf.materials.append(material)
                material_indices[wgeo_mesh.material] = material_index
            else:
                material_index = material_indices[wgeo_mesh.material]
            
            primitive = Primitive(
                attributes={
                    "POSITION": len(gltf.accessors) - 3,
                    "TEXCOORD_0": len(gltf.accessors) - 2
                },
                indices=len(gltf.accessors) - 1,
                material=material_index
            )
            
            mesh = Mesh(primitives=[primitive], name=f"{wgeo_mesh.material}")
            gltf.meshes.append(mesh)
            
            gltf.nodes.append(Node(mesh=len(gltf.meshes) - 1))
        
        buffer = Buffer(byteLength=len(buffer_data))
        gltf.buffers.append(buffer)
        
        output_glb = output_file.replace('.gltf', '.glb')
        gltf.set_binary_blob(buffer_data)
        gltf.save(output_glb)
        
        return gltf

def print_header():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                        WGEO to GLB Converter                     ║
║                          by: @jeanxpereira                       ║
║                                                                  ║
║     Convert WGEO (World Geometry) files to GLB (glTF Binary)     ║
╚══════════════════════════════════════════════════════════════════╝
    """)

def print_help():
    print("""
Usage: python WGEO.py [OPTIONS] <wgeo_file>

Options:
  -h, --help     Show this help message and exit
  -v, --version  Show program's version number and exit
  -o OUTPUT      Specify output file name (default: input file name with .glb extension)

Examples:
  python WGEO.py room.wgeo
  python WGEO.py -o custom_output.glb room.wgeo
    """)

def main():
    parser = argparse.ArgumentParser(description="Convert WGEO files to GLB", add_help=False)
    parser.add_argument('-h', '--help', action='store_true', help='Show this help message and exit')
    parser.add_argument('-v', '--version', action='version', version='WGEO to GLB Converter v1.0', help="Show program's version number and exit")
    parser.add_argument('-o', metavar='OUTPUT', help='Specify output file name')
    parser.add_argument('wgeo_file', nargs='?', help='Input WGEO file')

    args = parser.parse_args()

    if args.help or len(sys.argv) == 1:
        print_header()
        print_help()
        return

    if not args.wgeo_file:
        print("Error: No input file specified.")
        print("Use -h or --help for usage information.")
        return

    wgeo_file = args.wgeo_file
    output_file = args.o if args.o else os.path.splitext(wgeo_file)[0] + ".glb"

    print_header()
    print(f"Processing {wgeo_file}...")

    try:
        wgeo = WGEOReader.read_file(wgeo_file)
        mat_file = wgeo_file.replace('.wgeo', '.mat')
        if os.path.exists(mat_file):
            materials = MaterialReader.read_mat_file(mat_file)
        else:
            print(f"Warning: .mat file not found for {wgeo_file}")
            materials = {}
        
        gltf = GLTFExporter.export(wgeo, materials, output_file)
        
        print(f"\nSuccessfully converted {wgeo_file}")
        print(f"Number of meshes: {len(wgeo.meshes)}")
        print(f"Number of nodes in GLTF: {len(gltf.nodes)}")
        print(f"Number of meshes in GLTF: {len(gltf.meshes)}")
        print(f"Number of materials in GLTF: {len(gltf.materials)}")
        print(f"Number of textures in GLTF: {len(gltf.textures)}")
        print(f"Number of images in GLTF: {len(gltf.images)}")
        print(f"Number of accessors in GLTF: {len(gltf.accessors)}")
        print(f"Number of buffer views in GLTF: {len(gltf.bufferViews)}")
        
        for i, mesh in enumerate(wgeo.meshes):
            print(f"\nMesh {i + 1}:")
            print(f"  Texture: {mesh.texture}")
            print(f"  Material: {mesh.material}")
            print(f"  Vertices: {len(mesh.verts)}")
            print(f"  Indices: {len(mesh.inds)}")
        
        print(f"\nGLB file saved: {output_file}")
        print(f"GLB file size: {os.path.getsize(output_file)} bytes")
        
    except Exception as e:
        print(f"Error processing {wgeo_file}:")
        print(f"Type: {type(e).__name__}")
        print(f"Details: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()