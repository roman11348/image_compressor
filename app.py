
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from sklearn.cluster import KMeans

# -------------------- PAGE CONFIGURATION --------------------

st.set_page_config(
    page_title="Image Compressor",
    page_icon="🖼️",
    layout="wide"
)

# -------------------- FIXED COMPRESSION SETTINGS --------------------

MAX_ITERS = 10
TOLERANCE = 1e-3
USE_KMEANS_PLUS_PLUS = True
N_INIT = 1
ALGORITHM = "elkan"
RESIZE_IMAGE = True


# -------------------- IMAGE COMPRESSION --------------------

def compress_image_sklearn(
    image_array,
    K=8,
    max_iters=MAX_ITERS,
    tolerance=TOLERANCE,
    init_method="k-means++",
    n_init=N_INIT,
    algorithm=ALGORITHM
):
    """Compress an RGB image using K-Means clustering."""

    # Ensure image has three color channels
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)

    elif image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    elif image_array.shape[2] == 1:
        image_array = np.repeat(image_array, 3, axis=2)

    if image_array.shape[2] != 3:
        raise ValueError("Unsupported image format. Expected RGB.")

    # Convert every pixel into a row of RGB values
    height, width, channels = image_array.shape
    X_img = image_array.reshape(-1, channels)

    # K-Means clustering
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

    # Replace each pixel with its cluster centroid
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    X_recovered = centers[labels]

    # Restore image dimensions
    compressed_array = X_recovered.reshape(height, width, channels)

    # Convert pixel values to valid image values
    compressed_array = np.clip(
        compressed_array, 0, 255
    ).astype(np.uint8)

    converged = kmeans.n_iter_ < max_iters

    return compressed_array, kmeans.n_iter_, converged


def compress_image(image_array, K=8):
    """Use the fixed settings to compress an image."""

    init_method = (
        "k-means++" if USE_KMEANS_PLUS_PLUS else "random"
    )

    return compress_image_sklearn(
        image_array,
        K=K,
        max_iters=MAX_ITERS,
        tolerance=TOLERANCE,
        init_method=init_method,
        n_init=N_INIT,
        algorithm=ALGORITHM
    )


# -------------------- IMAGE RESIZING --------------------

def resize_image_half(image_array):
    """Resize an image to half its width and height."""

    height, width = image_array.shape[:2]

    new_height = max(1, height // 2)
    new_width = max(1, width // 2)

    img = Image.fromarray(image_array)

    resized_img = img.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    return np.array(resized_img)


# -------------------- DOWNLOAD FUNCTION --------------------

def create_download_data(image_array):
    """Convert the compressed image into downloadable PNG data."""

    img = Image.fromarray(image_array)
    buffer = io.BytesIO()

    img.save(buffer, format="PNG")

    return buffer.getvalue()


# -------------------- MAIN STREAMLIT APP --------------------

def main():

    st.title("🖼️ Image Compression Using K-Means Clustering")

    st.markdown(
        "Upload an image and reduce its number of colors "
        "using K-Means clustering."
    )

    # The only setting available to the user
    st.sidebar.header("Compression Parameters")

    K = st.sidebar.slider(
        "Number of Colors (K)",
        min_value=2,
        max_value=64,
        value=8,
        help="Lower values produce stronger color reduction."
    )

    # Image upload
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is None:
        st.info("👆 Upload an image to start compression.")
        return

    try:
        # Open and convert the uploaded image to RGB
        image = Image.open(uploaded_file).convert("RGB")
        original_array = np.array(image)

        original_height, original_width = original_array.shape[:2]

        # Count colors in the original image
        original_colors = len(
            np.unique(
                original_array.reshape(-1, 3),
                axis=0
            )
        )

        # Resize for faster processing
        if RESIZE_IMAGE:
            processing_array = resize_image_half(original_array)
        else:
            processing_array = original_array.copy()

        processing_height, processing_width = processing_array.shape[:2]

        # Display original image
        st.subheader("Image Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Original Image",
                use_container_width=True
            )

            st.write(
                f"**Dimensions:** {original_width} × "
                f"{original_height} pixels"
            )

            st.write(
                f"**Unique colors:** {original_colors:,}"
            )

        # Compress image
        with st.spinner("Compressing image using K-Means..."):

            compressed_array, iterations, converged = compress_image(
                processing_array,
                K=K
            )

        compressed_image = Image.fromarray(compressed_array)

        # Display compressed image
        with col2:
            st.image(
                compressed_image,
                caption=f"Compressed Image ({K} colors)",
                use_container_width=True
            )

            st.write(
                f"**Dimensions:** {processing_width} × "
                f"{processing_height} pixels"
            )

            st.write(
                f"**Unique colors:** "
                f"{len(np.unique(compressed_array.reshape(-1, 3), axis=0)):,}"
            )

        # Calculate color reduction
        compressed_colors = len(
            np.unique(
                compressed_array.reshape(-1, 3),
                axis=0
            )
        )

        color_reduction = (
            (1 - compressed_colors / original_colors) * 100
            if original_colors > 0 else 0
        )

        # Compression status
        if converged:
            st.success(
                f"Compression completed in {iterations} iterations."
            )
        else:
            st.warning(
                f"Compression finished after {iterations} iterations. "
                "The iteration limit may have been reached."
            )

        # Results
        st.subheader("Compression Results")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Colors Before",
            f"{original_colors:,}"
        )

        metric2.metric(
            "Colors After",
            f"{compressed_colors:,}"
        )

        metric3.metric(
            "Color Reduction",
            f"{color_reduction:.1f}%"
        )

        st.write(f"**K value:** {K}")
        st.write(f"**Iterations used:** {iterations}")

        # Download compressed image
        st.subheader("Download Compressed Image")

        download_data = create_download_data(compressed_array)

        st.download_button(
            label="⬇️ Download Compressed Image",
            data=download_data,
            file_name=f"compressed_{K}colors.png",
            mime="image/png"
        )

        # Side-by-side comparison plot
        st.subheader("Visual Comparison")

        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(12, 6)
        )

        ax1.imshow(original_array)
        ax1.set_title("Original Image")
        ax1.axis("off")

        ax2.imshow(compressed_array)
        ax2.set_title(f"Compressed Image (K={K})")
        ax2.axis("off")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if RESIZE_IMAGE:
            st.caption(
                "Note: The image is resized to half its width and height "
                "before compression to improve processing speed. "
                "The downloaded image uses these resized dimensions."
            )

    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.info("Please upload a valid PNG or JPEG image.")


if __name__ == "__main__":
    main()
