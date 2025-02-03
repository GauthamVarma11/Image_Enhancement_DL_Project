# Image Resolution and Denoising Enhancement using Deep Learning


## Overview
This project explores image restoration techniques using deep learning, specifically targeting image denoising and resolution enhancement. The primary goal is to reconstruct high-quality images from degraded versions, making them more visually appealing and information-rich.

## Key Contributions:
Image Denoising (Primary Focus) – Implemented U-Net and ResNet-inspired autoencoders to remove noise from images.
Image Resolution Enhancement – Developed and compared GAN and ESRGAN architectures for super-resolution.
Dataset – Used the MIT-Adobe FiveK dataset, which contains high-resolution images of various real-world scenes.
This project was part of an academic deep learning course (Semester 3) at University of Houston.

## Image Denoising using Autoencoders


📌 Why Image Denoising?
Image noise is a common problem in computer vision, caused by sensor limitations, low-light conditions, or transmission errors. Denoising is a crucial preprocessing step for applications in medical imaging, security, satellite imagery, and digital photography.

🔹 Approach
Two autoencoder-based architectures were implemented:

U-Net-Inspired Autoencoder – Optimized for pixel-level accuracy.
ResNet-Inspired Autoencoder – Focused on perceptual quality.

## Image Denoising Architecture
For image denoising, we implemented two autoencoder-based architectures: a U-Net-inspired autoencoder and a ResNet-inspired autoencoder. The U-Net model consists of an encoder-decoder structure with convolutional layers and skip connections, allowing the model to retain spatial information while progressively reducing noise. The encoder extracts features using convolutional layers followed by downsampling, and the decoder reconstructs the denoised image using upsampling layers. The final output layer maps pixel values to a normalized range using a sigmoid activation function.

The ResNet-inspired autoencoder takes a different approach by utilizing residual connections, which help in better gradient flow and prevent information loss. Instead of simple downsampling, residual blocks enable the model to learn both low-level and high-level features effectively. Unlike the U-Net, this model focuses on perceptual quality rather than strict pixel-wise reconstruction. It integrates perceptual loss (VGG16 features) to ensure that the denoised images are visually closer to real-world noise-free images. While the U-Net model performs well in minimizing mean squared error (MSE), the ResNet-based autoencoder produces more visually realistic results by focusing on high-level structures in images.


## Key Takeaways
U-Net autoencoder is best for pixel-accurate denoising (High PSNR, Low MSE).
ResNet autoencoder enhances perceptual quality, making denoised images look more natural.
ESRGAN improves image resolution but requires careful tuning to avoid artifacts.
GAN is good at realism but lacks super-resolution ability.
🚀 Future Enhancements
Implement Diffusion Models for more stable denoising.
Train on larger, real-world noisy datasets to improve model generalization.
Optimize ESRGAN’s discriminator to reduce artifacts while preserving fine details.
## Contributors
Gautham Sashi Nadimpalli – Image Denoising (U-Net, ResNet Autoencoder)
Tejas Murali – Image Resolution Enhancement (GAN, ESRGAN)
## References
U-Net Paper (Ronneberger et al., 2015)
ResNet Paper (He et al., 2016)
MIT-Adobe FiveK Dataset
## Final Thoughts
This project provides a comprehensive comparison of deep learning-based denoising and super-resolution techniques. It serves as a great foundation for future improvements using advanced architectures like Transformers or Diffusion Models.
