#importing libraries
import matplotlib
matplotlib.use('Tkagg')
from matplotlib import pyplot as plt 
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import numpy as np
import open3d as o3d

#Getting model
feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

#Loading and resizing image
Image = Image.open(r'D:\VS code\3D_model\input\input.jpg')
new_height = 480 if Image.height> 480 else Image.height
new_height -= (new_height%32)
new_width = int(new_height*Image.width/Image.height)
diff = new_width % 32

new_width = new_width - diff if diff < 16 else new_width + 32 - diff
new_size = (new_width,new_height)
Image = Image.resize(new_size)

#preparing image for the model
inputs = feature_extractor(images=Image, return_tensors="pt")

#Getting prediction from the model 
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

#Post processing
pad = 16
output = predicted_depth.squeeze().cpu().numpy()*1000.0    
output = output[pad:-pad, pad:-pad]
Image = Image.crop((pad, pad, Image.width - pad, Image.height - pad))

#Visulaise the prediction
fig, ax = plt.subplots(1,2)
ax[0].imshow(Image)
ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax[1].imshow(output, cmap='plasma')
ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()
plt.show()

#preparing depth image for open3d
width,height=Image.size
depth_image=(output*255/np.max(output)).astype('uint8')
Image=np.array(Image)

#create egbd image
depth_o3d = o3d.geometry.Image(depth_image)
Image_o3d = o3d.geometry.Image(Image)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(Image_o3d,depth_o3d, convert_rgb_to_intensity=False)

# creating camera
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(width,height,500,500,width/2,height/2)

#creating o3d point cloud
pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,camera_intrinsic)
o3d.visualization.draw_geometries([pcd_raw])

#post processing in o3d point cloud 
# outlier rremoval
cl,ind = pcd_raw.remove_statistical_outlier(nb_neighbors=20, std_ratio=20)
pcd = pcd_raw.select_by_index(ind)

#estimate normals
pcd.estimate_normals()
pcd.orient_normals_to_align_with_direction()
o3d.visualization.draw_geometries([pcd])

#surface recontruction
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,depth=10,n_threads=1)[0]

#rotate mesh
rotation = mesh.get_rotation_matrix_from_xyz((np.pi,0,0))
mesh.rotate(rotation, center=(0,0,0))

#visualze the mesh
o3d.visualization.draw_geometries([mesh],mesh_show_back_face=True)

##D mesh export 
o3d.io.write_triangle_mesh(r'D:\VS code\3D_model\output\3doutput.obj', mesh)