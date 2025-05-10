# Photo-to-3D Model Generator (Prototype)

This project is a Python-based prototype that generates a simple 3D model from either a photo input (camera or AI-generated image). It uses deep learning to estimate depth and reconstructs a basic 3D mesh.

---

## Folder Structure

input/               # Input images (e.g., inputai.jpg)  
output/              # Generated 3D .obj files  
3D_model.py              # Main Python script  
requirements.txt     # Dependencies  
README.md            # This file  

---

## How to Run the Project

1. Clone the Repository

   git clone https://github.com/yourusername/photo-to-3d-model.git  
   cd photo-to-3d-model  

2. Create a Virtual Environment

   python -m venv venv  
   venv\Scripts\activate  (on Windows)  
   source venv/bin/activate  (on macOS/Linux)

3. Install Dependencies

   pip install -r requirements.txt

4. Run the Script

   python main.py

The script reads an image from `input/`, performs depth estimation, reconstructs a point cloud and mesh, and saves a `.obj` file in `output/`.

---

## Thought Process

- I used the GLPN model (from Hugging Face) for monocular depth estimation.
- The output depth map is used to build a point cloud using Open3D.
- A Poisson mesh reconstruction algorithm converts the point cloud into a 3D mesh.
- The mesh is saved in .obj format, suitable for loading in 3D software or viewers.

---

## Libraries Used

- torch  
- transformers (GLPNForDepthEstimation)  
- Pillow  
- matplotlib  
- numpy  
- open3d  

---

## Notes

- Works best on images with a clear, centered object and minimal background clutter.
- You can swap in different images by placing them in the `input/` folder.

---

## Output

- Visual previews of:
  - Input image
  - Depth map
  - Point cloud
  - Final 3D mesh
- `.obj` file saved in the `output/` directory.

