# DIP-Bilateral-Filter
Digital Image Processing

In this code, 
OpenCV’s 3x3 box filter, 
OpenCV’s 5x5 box filter, 
OpenCV’s 3x3 Gaussian filter (with auto var σ = 0), 
OpenCV’s 5x5 Gaussian filter (with auto var σ = 0), 
Adaptive mean filter (5x5 and assume 𝜎= 0.0042),
OpenCV’s bilateral filter cv.bilateralFilter(inputImg, 5, 3, 0.9, borderType = cv.BORDER_REPLICATE) are applied to image(noisyImage_Gaussian.jpg) and their PSNR values are compared. (as a ground truth image, (lena_grayscale_hq.jpg) is used for PSNR) 

After these operations, same operators are applied to (noisyImage_Gaussian_01.jpg) for checking the performances of filters with different noise level. Additionally, their PSNR values are calculated and compared. 

In the last part, Bilateral filter is implamented and compared with OpenCV bilateral filter output. 
