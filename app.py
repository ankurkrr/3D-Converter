import streamlit as st
import open3d as o3d
import numpy as np
import torch
import tempfile
import os
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.graph_objects as go

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

from transformers import GLPNImageProcessor, GLPNForDepthEstimation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_text_to_3d_models():
    """Load the text-to-3D models"""
    base_name = 'base40M-textvec'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    base_model.load_state_dict(load_checkpoint(base_name, device))
    
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))
    
    return base_model, base_diffusion, upsampler_model, upsampler_diffusion

@st.cache_resource
def load_image_to_3d_models():
    """Load the image-to-3D models"""
    feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
    return feature_extractor, model

def text_to_3d(prompt):
    """Convert text to 3D model"""
    base_model, base_diffusion, upsampler_model, upsampler_diffusion = load_text_to_3d_models()
    
    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=('texts', '')
    )
    
    st.write(f"Generating 3D model for: '{prompt}'")
    progress_bar = st.progress(0)
    samples = None
    
    for i, x in enumerate(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
        samples = x
        progress_bar.progress(min(1.0, (i + 1) / 20))
    
    pc = sampler.output_to_point_clouds(samples)[0]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.coords)
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.03)
    
    vertex_colors = np.tile(np.array([0.0, 0.0, 1.0]), (len(mesh.vertices), 1))
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh.compute_vertex_normals()
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.obj', delete=False)
    o3d.io.write_triangle_mesh(temp_file.name, mesh)
    
    return temp_file.name, pc

def image_to_3d(image):
    """Convert image to 3D model"""
    feature_extractor, model = load_image_to_3d_models()
    
    new_height = 480 if image.height > 480 else image.height
    new_width = int(new_height * image.width / image.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + 32 - diff
    new_size = (new_width, new_height)
    image = image.resize(new_size)
    
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    pad = 16
    output = predicted_depth.squeeze().cpu().numpy() * 1000.0
    output = output[pad:-pad, pad:-pad]
    image = image.crop((pad, pad, image.width - pad, image.height - pad))
    
    width, height = image.size
    depth_image = (output * 255 / np.max(output)).astype('uint8')
    image_np = np.array(image)
    
    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(image_np)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image_o3d, depth_o3d, convert_rgb_to_intensity=False)
    
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, 500, 500, width/2, height/2)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)
    pcd = pcd.select_by_index(ind)
    
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads=1)[0]
    
    rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(rotation, center=(0, 0, 0))
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.obj', delete=False)
    o3d.io.write_triangle_mesh(temp_file.name, mesh)
    
    return temp_file.name, pcd

def visualize_point_cloud(pc):
    """Create an interactive 3D plot of the point cloud using Plotly"""
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pc.coords[:, 0],
                y=pc.coords[:, 1],
                z=pc.coords[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='blue',
                    opacity=0.8
                )
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig

def main():
    st.title("2D to 3D Conversion App")
    st.write("Convert text descriptions or images to 3D models")

    tab1, tab2 = st.tabs(["Text to 3D", "Image to 3D"])

    with tab1:
        st.header("Text to 3D")
        prompt = st.text_input("Enter a description:", "a blue car")

        if st.button("Generate 3D from Text"):
            with st.spinner("Generating 3D model..."):
                obj_file, pc = text_to_3d(prompt)

                st.subheader("Point Cloud Preview")
                preview_img = visualize_point_cloud(pc)
                st.plotly_chart(preview_img)

                with open(obj_file, "rb") as file:
                    st.download_button(
                        label="Download 3D Model (.obj)",
                        data=file,
                        file_name=f"{prompt.replace(' ', '_')}.obj",
                        mime="application/octet-stream"
                    )

    with tab2:
        st.header("Image to 3D")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Generate 3D from Image"):
                with st.spinner("Generating 3D model..."):
                    obj_file, pcd = image_to_3d(image)

                    st.subheader("Point Cloud Preview")
                    points = np.asarray(pcd.points)
                    colors = np.asarray(pcd.colors)

                    fig = go.Figure(
                        data=[
                            go.Scatter3d(
                                x=points[:, 0],
                                y=points[:, 1],
                                z=points[:, 2],
                                mode='markers',
                                marker=dict(
                                    size=2,
                                    color=colors,
                                    opacity=0.8
                                )
                            )
                        ]
                    )

                    fig.update_layout(
                        scene=dict(
                            xaxis_title='X',
                            yaxis_title='Y',
                            zaxis_title='Z'
                        ),
                        margin=dict(l=0, r=0, t=0, b=0)
                    )

                    st.plotly_chart(fig)

                    with open(obj_file, "rb") as file:
                        st.download_button(
                            label="Download 3D Model (.obj)",
                            data=file,
                            file_name=f"image_to_3d.obj",
                            mime="application/octet-stream"
                        )

if __name__ == "__main__":
    main()