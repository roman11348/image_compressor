import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Image Compression using K-Means",
    page_icon="🖼️",
    layout="wide"
)

# K-means clustering using scikit-learn
def compress_image_sklearn(image_array, K=16, max_iters=10, tolerance=1e-4, init_method='k-means++', 
                          n_init=1, algorithm='auto', progress_callback=None):
    """
    Compress an image using scikit-learn's KMeans clustering
    
    Args:
        image_array: numpy array of the image
        K: number of colors to compress to
        max_iters: maximum number of iterations
        tolerance: convergence tolerance
        init_method: initialization method ('k-means++' or 'random')
        n_init: number of initializations
        algorithm: K-means algorithm ('auto', 'full', 'elkan')
        progress_callback: optional progress callback
    
    Returns:
        compressed_image: compressed image array
        iterations: number of iterations used
        converged: whether algorithm converged
    """
    
    if len(image_array.shape) == 2:
        
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:
       
        image_array = image_array[:, :, :3]
    elif image_array.shape[2] == 1:
        # Single channel - convert to RGB
        image_array = np.repeat(image_array, 3, axis=2)
    
    
    if image_array.shape[2] != 3:
        raise ValueError(f"Unsupported image format with {image_array.shape[2]} channels. Expected RGB (3 channels).")
    
    # Reshape image to 2D array where each row is a pixel
    X_img = np.reshape(image_array, (image_array.shape[0] * image_array.shape[1], 3))
    
    # Create KMeans model with scikit-learn
    kmeans = KMeans(
        n_clusters=K,
        max_iter=max_iters,
        tol=tolerance,
        init=init_method,
        random_state=42,
        n_init=n_init,
        algorithm=algorithm
    )
    
    
    kmeans.fit(X_img)
    
    
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    
    X_recovered = centers[labels]
    
    # Reshape image into proper dimensions
    X_recovered = np.reshape(X_recovered, image_array.shape)
    
    converged = kmeans.n_iter_ < max_iters
    
    return X_recovered.astype(np.uint8), kmeans.n_iter_, converged

def compress_image(image_array, K=16, max_iters=10, tolerance=1e-4, use_kmeans_plus_plus=True, 
                  n_init=1, algorithm='full', progress_callback=None):

    # Choose initialization method
    init_method = 'k-means++' if use_kmeans_plus_plus else 'random'
    
    # Use scikit-learn implementation
    return compress_image_sklearn(image_array, K, max_iters, tolerance, init_method, n_init, algorithm, progress_callback)

def resize_image_half(image_array):
    """
    Resize image to half size while maintaining aspect ratio
    
    Args:
        image_array: numpy array of the image
    
    Returns:
        resized_image_array: resized image array
    """
    height, width = image_array.shape[:2]
    new_height = height // 2
    new_width = width // 2
    
    # Use PIL for high-quality resizing
    img = Image.fromarray(image_array)
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return np.array(resized_img)

def create_download_link(image_array, filename="compressed_image.png"):
    """Create a download link for the compressed image"""
    img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()

# Streamlit UI
def main():
    st.title("🖼️ Image Compression using K-Means Clustering")
    st.markdown("Upload an image and compress it by reducing the number of colors using scikit-learn's optimized K-means clustering.")
    
    # Sidebar for parameters
    st.sidebar.header("Compression Parameters")
    K = st.sidebar.slider("Number of Colors (K)", min_value=2, max_value=64, value=8, 
                          help="Number of colors to compress the image to. Lower values = more compression")
    max_iters = st.sidebar.slider("Max Iterations", min_value=1, max_value=50, value=5,
                                 help="Maximum number of K-means iterations")
    tolerance = st.sidebar.slider("Convergence Tolerance", min_value=1e-6, max_value=1e-2, value=1e-3, format="%.0e",
                                 help="Stop early if centroid movement is below this threshold")
    
    # Optimization settings
    st.sidebar.header("Optimization Settings")
    use_kmeans_plus_plus = st.sidebar.checkbox("Use K-means++ Initialization", value=True,
                                              help="Better initialization for faster convergence (scikit-learn default)")
    
    # Additional scikit-learn options
    st.sidebar.header("Advanced Options")
    n_init = st.sidebar.slider("Number of Initializations", min_value=1, max_value=10, value=1,
                               help="Number of times K-means will be run with different centroid seeds")
    
    # Fixed settings for speed optimization
    algorithm = "elkan"  # Always use elkan for faster convergence
    resize_image = True  # Always resize to half size for faster processing
    
    # File upload
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            
            image = Image.open(uploaded_file)
            
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_array = np.array(image)
            
            # Resize image if requested
            original_size = image_array.shape[:2]
            if resize_image:
                image_array = resize_image_half(image_array)
                resized_size = image_array.shape[:2]
            
            st.subheader("Original Image")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
                st.write(f"**Original size:** {original_size[0]} x {original_size[1]} pixels")
                if resize_image:
                    st.write(f"**Resized for processing:** {resized_size[0]} x {resized_size[1]} pixels")
                st.write(f"**Image format:** {image.mode} → RGB")
                st.write(f"**Original colors:** {len(np.unique(image_array.reshape(-1, image_array.shape[2]), axis=0))} unique colors")
            
            # Compress image
            with st.spinner("Compressing image using scikit-learn K-means clustering..."):
                compressed_array, iterations, converged = compress_image(
                    image_array, K=K, max_iters=max_iters, tolerance=tolerance, 
                    use_kmeans_plus_plus=use_kmeans_plus_plus, n_init=n_init, algorithm=algorithm
                )
                compressed_image = Image.fromarray(compressed_array)
            
            with col2:
                st.image(compressed_image, caption=f"Compressed Image ({K} colors)", use_container_width=True)
                if resize_image:
                    st.write(f"**Compressed size:** {compressed_array.shape[0]} x {compressed_array.shape[1]} pixels (resized)")
                    st.write(f"**Original size:** {original_size[0]} x {original_size[1]} pixels")
                else:
                    st.write(f"**Compressed size:** {compressed_array.shape[0]} x {compressed_array.shape[1]} pixels")
                st.write(f"**Compressed colors:** {len(np.unique(compressed_array.reshape(-1, compressed_array.shape[2]), axis=0))} unique colors")
            
            # Calculate compression ratio
            original_colors = len(np.unique(image_array.reshape(-1, image_array.shape[2]), axis=0))
            compressed_colors = len(np.unique(compressed_array.reshape(-1, compressed_array.shape[2]), axis=0))
            compression_ratio = (1 - compressed_colors / original_colors) * 100
            
            # Show compression results
            if converged:
                st.success(f"✅ Compression completed in {iterations} iterations (converged)! Color reduction: {compression_ratio:.1f}%")
            else:
                st.warning(f"⚠️ Compression completed in {iterations} iterations (max iterations reached). Color reduction: {compression_ratio:.1f}%")
            
            # Show optimization info
            with st.expander("Compression Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Iterations Used", iterations)
                with col2:
                    st.metric("Converged", "Yes" if converged else "No")
                with col3:
                    st.metric("Color Reduction", f"{compression_ratio:.1f}%")
            
            # Download button
            st.subheader("Download Compressed Image")
            download_data = create_download_link(compressed_array)
            st.download_button(
                label="Download Compressed Image",
                data=download_data,
                file_name=f"compressed_{K}colors_{uploaded_file.name}",
                mime="image/png"
            )
            
            # Show comparison
            st.subheader("Side-by-Side Comparison")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            ax1.imshow(image_array)
            ax1.set_title(f"Original ({original_colors} colors)")
            ax1.axis('off')
            
            ax2.imshow(compressed_array)
            ax2.set_title(f"Compressed ({compressed_colors} colors)")
            ax2.axis('off')
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.write("Please make sure you've uploaded a valid image file.")
    
    else:
        st.info("👆 Please upload an image file to get started!")
        
        # Show example with test images if they exist
        if os.path.exists("test.jpg") or os.path.exists("test.png"):
            st.subheader("Example with Test Images")
            
            test_files = []
            if os.path.exists("test.jpg"):
                test_files.append("test.jpg")
            if os.path.exists("test.png"):
                test_files.append("test.png")
            
            for test_file in test_files:
                if st.button(f"Process {test_file}"):
                    try:
                        image = Image.open(test_file)
                        
                        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        image_array = np.array(image)
                        
                        # Resize image if requested
                        original_size = image_array.shape[:2]
                        if resize_image:
                            image_array = resize_image_half(image_array)
                            resized_size = image_array.shape[:2]
                        
                        st.subheader(f"Processing {test_file}")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.image(image, caption="Original Image", use_container_width=True)
                            st.write(f"**Original size:** {original_size[0]} x {original_size[1]} pixels")
                            if resize_image:
                                st.write(f"**Resized for processing:** {resized_size[0]} x {resized_size[1]} pixels")
                        
                        with st.spinner("Compressing image..."):
                            compressed_array, iterations, converged = compress_image(
                                image_array, K=K, max_iters=max_iters, tolerance=tolerance,
                                use_kmeans_plus_plus=use_kmeans_plus_plus, n_init=n_init, algorithm=algorithm
                            )
                            compressed_image = Image.fromarray(compressed_array)
                        
                        with col2:
                            st.image(compressed_image, caption=f"Compressed Image ({K} colors)", use_container_width=True)
                        
                        # Download button for test image
                        download_data = create_download_link(compressed_array)
                        st.download_button(
                            label=f"Download Compressed {test_file}",
                            data=download_data,
                            file_name=f"compressed_{K}colors_{test_file}",
                            mime="image/png"
                        )
                        
                    except Exception as e:
                        st.error(f"Error processing {test_file}: {str(e)}")

if __name__ == "__main__":
    main()