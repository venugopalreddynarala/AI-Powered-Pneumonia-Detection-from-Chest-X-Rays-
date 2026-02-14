"""
3D lung visualization with damage overlay.
Maps 2D Grad-CAM heatmaps to 3D lung surface for interactive visualization.
"""

import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Tuple, Optional
import cv2


def create_simple_3d_lungs(heatmap: np.ndarray, 
                          severity: str) -> go.Figure:
    """
    Create 3D lung visualization with color-coded severity.
    Since we may not have the actual .glb file, we create a parametric lung shape.
    
    Args:
        heatmap: Grad-CAM heatmap (H, W) normalized [0, 1]
        severity: Severity level (Mild/Moderate/Severe)
        
    Returns:
        Plotly Figure object
    """
    # Calculate average intensity for overall color
    avg_intensity = np.mean(heatmap)
    
    # Map intensity to color
    if avg_intensity < 0.3:
        color = 'green'
        severity_text = 'Low Risk'
    elif avg_intensity < 0.6:
        color = 'yellow'
        severity_text = 'Moderate Risk'
    else:
        color = 'red'
        severity_text = 'High Risk'
    
    # Create parametric lung shapes (simplified)
    # Right lung
    u_right = np.linspace(0, 2 * np.pi, 50)
    v_right = np.linspace(0, np.pi, 50)
    u_right, v_right = np.meshgrid(u_right, v_right)
    
    x_right = 2 + 1.5 * np.sin(v_right) * np.cos(u_right)
    y_right = 1.5 * np.sin(v_right) * np.sin(u_right)
    z_right = 2.5 * np.cos(v_right)
    
    # Left lung (mirrored)
    x_left = -2 - 1.5 * np.sin(v_right) * np.cos(u_right)
    y_left = y_right
    z_left = z_right
    
    # Map heatmap intensity to colors
    colors_right = get_heatmap_colors(heatmap, x_right.shape)
    colors_left = get_heatmap_colors(heatmap, x_left.shape)
    
    # Create 3D surface plots
    fig = go.Figure()
    
    # Right lung
    fig.add_trace(go.Surface(
        x=x_right, y=y_right, z=z_right,
        surfacecolor=colors_right,
        colorscale=[[0, 'rgb(0,255,0)'], [0.5, 'rgb(255,255,0)'], [1, 'rgb(255,0,0)']],
        showscale=True,
        name='Right Lung',
        hovertemplate='<b>Right Lung</b><br>Intensity: %{surfacecolor:.2f}<extra></extra>'
    ))
    
    # Left lung
    fig.add_trace(go.Surface(
        x=x_left, y=y_left, z=z_left,
        surfacecolor=colors_left,
        colorscale=[[0, 'rgb(0,255,0)'], [0.5, 'rgb(255,255,0)'], [1, 'rgb(255,0,0)']],
        showscale=False,
        name='Left Lung',
        hovertemplate='<b>Left Lung</b><br>Intensity: %{surfacecolor:.2f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'3D Lung Visualization - {severity}<br>Average Intensity: {avg_intensity:.2%}',
            x=0.5,
            xanchor='center',
            font=dict(size=16, color='white')
        ),
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title=''),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
            bgcolor='rgb(17, 17, 17)'
        ),
        paper_bgcolor='rgb(17, 17, 17)',
        plot_bgcolor='rgb(17, 17, 17)',
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


def get_heatmap_colors(heatmap: np.ndarray, target_shape: Tuple) -> np.ndarray:
    """
    Resize and map heatmap to target shape for 3D coloring.
    
    Args:
        heatmap: Original heatmap (H, W)
        target_shape: Target shape for 3D mesh
        
    Returns:
        Resized heatmap matching target shape
    """
    # Resize heatmap to match mesh dimensions
    resized = cv2.resize(heatmap, (target_shape[1], target_shape[0]))
    
    # Add some randomness for realistic variation
    noise = np.random.normal(0, 0.05, resized.shape)
    resized = np.clip(resized + noise, 0, 1)
    
    return resized


def create_damage_overlay_3d(heatmap: np.ndarray,
                            confidence: float,
                            severity: str) -> go.Figure:
    """
    Create enhanced 3D visualization with damage regions highlighted.
    
    Args:
        heatmap: Grad-CAM heatmap
        confidence: Model confidence
        severity: Severity classification
        
    Returns:
        Plotly Figure
    """
    fig = create_simple_3d_lungs(heatmap, severity)
    
    # Add annotation with statistics
    affected_percentage = (heatmap > 0.5).sum() / heatmap.size * 100
    
    annotation_text = (
        f"<b>Analysis Results</b><br>"
        f"Severity: {severity}<br>"
        f"Confidence: {confidence:.1%}<br>"
        f"Affected Area: {affected_percentage:.1f}%"
    )
    
    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="rgba(0, 0, 0, 0.7)",
        font=dict(color="white", size=12),
        align="left",
        bordercolor="white",
        borderwidth=1
    )
    
    return fig


def load_3d_lung_model(glb_path: str) -> Optional[Dict]:
    """
    Load 3D lung model from GLB file if available.
    
    Args:
        glb_path: Path to .glb file
        
    Returns:
        Dictionary with mesh data or None if not found
    """
    try:
        import trimesh
        
        mesh = trimesh.load(glb_path)
        
        return {
            'vertices': mesh.vertices,
            'faces': mesh.faces,
            'normals': mesh.vertex_normals if hasattr(mesh, 'vertex_normals') else None
        }
    except Exception as e:
        print(f"Could not load 3D model: {e}")
        print("Using parametric lung model instead")
        return None


def map_heatmap_to_3d_mesh(heatmap: np.ndarray, 
                          mesh_data: Dict) -> np.ndarray:
    """
    Map 2D heatmap intensity to 3D mesh vertices.
    
    Args:
        heatmap: 2D Grad-CAM heatmap
        mesh_data: Dictionary with vertices and faces
        
    Returns:
        Color values for each vertex
    """
    vertices = mesh_data['vertices']
    
    # Normalize vertex positions to [0, 1] range
    x_norm = (vertices[:, 0] - vertices[:, 0].min()) / (vertices[:, 0].max() - vertices[:, 0].min())
    y_norm = (vertices[:, 1] - vertices[:, 1].min()) / (vertices[:, 1].max() - vertices[:, 1].min())
    
    # Map to heatmap coordinates
    h, w = heatmap.shape
    x_idx = (x_norm * (w - 1)).astype(int)
    y_idx = (y_norm * (h - 1)).astype(int)
    
    # Get color values from heatmap
    vertex_colors = heatmap[y_idx, x_idx]
    
    return vertex_colors


def create_plotly_3d_from_mesh(mesh_data: Dict,
                               vertex_colors: np.ndarray,
                               title: str = "3D Lung Model") -> go.Figure:
    """
    Create Plotly 3D mesh from mesh data with vertex colors.
    
    Args:
        mesh_data: Dictionary with vertices and faces
        vertex_colors: Color value for each vertex
        title: Figure title
        
    Returns:
        Plotly Figure
    """
    vertices = mesh_data['vertices']
    faces = mesh_data['faces']
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=vertex_colors,
            colorscale=[[0, 'green'], [0.5, 'yellow'], [1, 'red']],
            showscale=True,
            hovertemplate='<b>Lung Surface</b><br>Intensity: %{intensity:.2f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        height=600
    )
    
    return fig


def create_side_by_side_view(original_img: np.ndarray,
                             heatmap_overlay: np.ndarray,
                             fig_3d: go.Figure) -> Dict:
    """
    Create comprehensive visualization with 2D and 3D views.
    
    Args:
        original_img: Original X-ray image
        heatmap_overlay: Heatmap overlaid on X-ray
        fig_3d: 3D Plotly figure
        
    Returns:
        Dictionary with all visualization components
    """
    return {
        'original': original_img,
        'heatmap_overlay': heatmap_overlay,
        '3d_figure': fig_3d
    }


def export_3d_html(fig: go.Figure, output_path: str = "lung_3d.html"):
    """
    Export 3D visualization to standalone HTML file.
    
    Args:
        fig: Plotly figure
        output_path: Output HTML file path
    """
    fig.write_html(output_path)
    print(f"3D visualization saved to {output_path}")


if __name__ == "__main__":
    # Test 3D visualization
    print("Testing 3D visualization module...")
    
    # Create dummy heatmap
    heatmap = np.random.rand(224, 224)
    heatmap[50:150, 50:150] = 0.8  # High intensity region
    
    # Create visualization
    fig = create_simple_3d_lungs(heatmap, "Moderate")
    
    print("✓ 3D visualization created successfully!")
    print("  Use fig.show() to display or fig.write_html('output.html') to save")
