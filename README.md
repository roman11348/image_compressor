# 🖼️ Image Compression using K-Means Clustering

A Streamlit web application that compresses images by reducing the number of colors using scikit-learn's optimized K-means clustering algorithm.

## ✨ Features

- **Smart Image Processing**: Handles RGB, RGBA, and grayscale images automatically
- **Interactive UI**: Real-time parameter adjustment with sliders and controls
- **Advanced K-means Options**: 
  - K-means++ initialization for better convergence
  - Multiple algorithm choices (auto, full, elkan)
  - Configurable convergence tolerance
  - Multiple initialization runs
- **Visual Comparison**: Side-by-side original vs compressed image display
- **Download Results**: Save compressed images directly from the app
- **Performance Metrics**: Shows compression ratio, iterations used, and convergence status

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   streamlit run app.py
   ```

3. **Upload an image** and adjust compression parameters in the sidebar

## 📊 Parameters

- **Number of Colors (K)**: 2-64 colors (lower = more compression)
- **Max Iterations**: 1-50 iterations
- **Convergence Tolerance**: Early stopping threshold
- **Initialization Method**: K-means++ (recommended) or random
- **Algorithm**: auto, full, or elkan K-means variants
- **Number of Initializations**: Multiple runs for better results

## 🎯 Next Goals: Performance Optimizations

The next development phase will focus on advanced optimizations:

- **GPU Acceleration**: CUDA support for faster processing
- **Parallel Processing**: Multi-threading for large images
- **Memory Optimization**: Efficient handling of very large images
- **Batch Processing**: Compress multiple images simultaneously
- **Advanced Algorithms**: Mini-batch K-means and other variants
- **Real-time Preview**: Live compression preview as parameters change
- **Format Support**: Additional image formats (WebP, TIFF, etc.)
- **Quality Metrics**: PSNR, SSIM, and other quality assessments

## 📁 Files

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `model.ipynb` - Original Jupyter notebook with K-means implementation
- `image.py` - Image format utilities
- `test.jpg`, `test.png` - Sample test images

## 🔧 Dependencies

- streamlit
- scikit-learn
- numpy
- matplotlib
- Pillow (PIL)

---

*Built with ❤️ using Streamlit and scikit-learn*