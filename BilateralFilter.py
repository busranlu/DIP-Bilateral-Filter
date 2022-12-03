#Busra_Unlu_211711008_HW7

import numpy as np
import cv2 as cv

#load image
image = cv.imread("C:/Users/Busra/Desktop/BIL561_Odev/hw7/noisyImage_Gaussian.jpg",0)
image2=cv.imread("C:/Users/Busra/Desktop/BIL561_Odev/hw7/noisyImage_Gaussian_01.jpg",0)
groundTruth = cv.imread('C:/Users/Busra/Desktop/BIL561_Odev/hw7/lena_grayscale_hq.jpg', 0)

#normalise image [0-1]
normalised_image= cv.normalize(image, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
normalised_image2= cv.normalize(image2, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)


#QUESTION 1
def adaptiveMeanFilter(normalised_image,Sxy,noise_variance):   
    (h, w) = normalised_image.shape[:2]
    local_average=np.zeros((h,w))
    mean_squared_image=np.zeros((h,w))
    f=np.zeros((h,w))
    #padding
    top=left=right=bottom=(Sxy-1)//2
    padded_image = cv.copyMakeBorder( normalised_image,top, bottom, left, right, cv.BORDER_REPLICATE, None, value = 0 )
    #finding mean image
    for i in range(w):
        for j in range(h):
            temp = padded_image[i:i + Sxy, j:j + Sxy]
            local_average[i,j]=np.mean(temp)   
    
    #square of mean image
    square_mean_image=np.square(local_average)

    #square of image
    squared_image=np.square(padded_image)

    #mean of squared image
    for i in range(w):
        for j in range(h):
            temp2 = squared_image[i:i + Sxy, j:j + Sxy]
            mean_squared_image[i,j]=np.mean(temp2)

    #local variance
    local_variance= mean_squared_image - square_mean_image

    #formule implamentation
    f=normalised_image - ((noise_variance / local_variance) * (normalised_image - local_average ) )
    
    #get back[0-255]
    f = cv.normalize(f, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    return f.astype(np.uint8)

#OpenCV's 3x3 box filter.
output_1_1=cv.blur(image, (3,3), borderType = cv.BORDER_REPLICATE)
#output_1_1 = cv.normalize(output_1_1, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

#ii. OpenCV's 5x5 box filter.
output_1_2=cv.blur(image, (5,5), borderType = cv.BORDER_REPLICATE)
#output_1_2 = cv.normalize(output_1_2, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

#iii. OpenCV''s 3x3 Gaussian filter (Ïƒ = 0).
output_1_3=cv.GaussianBlur(image, (3, 3), 0, borderType = cv.BORDER_REPLICATE)
#output_1_3 = cv.normalize(output_1_3, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

#iv. OpenCV's 5x5 Gaussian filter (Ïƒ = 0).
output_1_4=cv.GaussianBlur(image, (5, 5), 0, borderType = cv.BORDER_REPLICATE)
#output_1_4 = cv.normalize(output_1_4, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

#v. Your adaptive mean filter from HW#5 (5x5 and assume ðœŽ= 0.0042).
output_1_5=adaptiveMeanFilter(normalised_image,5,0.0042)

#vi. OpenCV's bilateral filter as:
bilateral_OpenCV = cv.bilateralFilter(image, 5, 3, 0.9, borderType = cv.BORDER_REPLICATE)
#bilateral_OpenCV = cv.normalize(bilateral_OpenCV, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)


#QUESTION 2
#OpenCV's 3x3 box filter.
output_2_1=cv.blur(image2, (3,3), borderType = cv.BORDER_REPLICATE)
#output_2_1 = cv.normalize(output_2_1, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

#ii. OpenCV's 5x5 box filter.
output_2_2=cv.blur(image2, (5,5), borderType = cv.BORDER_REPLICATE)
#output_2_2 = cv.normalize(output_2_2, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

#iii. OpenCV's 3x3 Gaussian filter (with auto var ïƒ¨ Ïƒ = 0).
output_2_3=cv.GaussianBlur(image2, (3, 3), 0, borderType = cv.BORDER_REPLICATE)
#output_2_3 = cv.normalize(output_2_3, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

#iv. OpenCV's 5x5 Gaussian filter (with auto var ïƒ¨ Ïƒ = 0).
output_2_4=cv.GaussianBlur(image2, (5, 5), 0, borderType = cv.BORDER_REPLICATE)
#output_2_4 = cv.normalize(output_2_4, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

#v. Your adaptive mean filter from HW#5 (5x5 and assume ðœŽà°Žà¬¶ = 0.0042).
output_2_5=adaptiveMeanFilter(normalised_image2,5,0.0009)

#vi. OpenCV's bilateral filter as:
bilateral_OpenCV_2 = cv.bilateralFilter(image2, 3, 0.1, 1, borderType = cv.BORDER_REPLICATE)
#bilateral_OpenCV_2 = cv.normalize(bilateral_OpenCV_2, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)


#QUESTION 3
#my bilateral filter:
def bilateralFilter(image,sigma_s,sigma_i,kernel_size):

    #padding
    top=bottom=left=right=kernel_size//2
    padImg = cv.copyMakeBorder( image, top, bottom, left, right, cv.BORDER_REPLICATE)

    #normalise image [0-1]
    #normalised_image= cv.normalize(padImg, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    
    normalised_image = padImg / 255
    normalised_image = normalised_image.astype("float32")
    w,h=normalised_image.shape

    output_image = np.zeros(normalised_image.shape)

    # gaussian kernel
    arr = np.zeros((kernel_size, kernel_size))
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            arr[i, j] = np.sqrt(abs(i - kernel_size // 2) ** 2 + abs(j - kernel_size // 2) ** 2)
    sigma = np.sqrt(sigma_s)
    cons = 1 / (sigma * np.sqrt(2 * np.pi))
    gauss_kernel= cons * np.exp(-((arr / sigma) ** 2) * 0.5)
    

    for i in range(top, w - top):
        for j in range(top, h - top):
 
            imgS=normalised_image[i - top : i + top + 1, j - top : j + top + 1]
            
            imgI = imgS - imgS[top, top]

            #apply gaussian kernel
            sigma = np.sqrt(sigma_i)
            cons = 1 / (sigma * np.sqrt(2 * np.pi))
            imgIG = cons * np.exp(-((imgI / sigma) ** 2) * 0.5)


            weights = np.multiply(gauss_kernel, imgIG)
            temps = np.multiply(imgS, weights)
            temp = np.sum(temps) / np.sum(weights)
            output_image[i, j] = temp

    #get back[0-255]
    #output_image = cv.normalize(output_image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    output_image = output_image * 255
    #out = np.ceil(out - 0.5)
    output_image = np.uint8(output_image)

    return output_image


bilateral_out = bilateralFilter(image, 0.9 , 3 , 5)
w,h=bilateral_out.shape
padding = 5//2
bilateral_out = bilateral_out[padding:w-padding, padding:h-padding]

bilateral_out_2 = bilateralFilter(image, 1 , 0.3 , 3)
bilateral_out_2 = bilateral_out_2[padding:w-padding, padding:h-padding]


#PSNR
#Q1
psnr_1_1=cv.PSNR(groundTruth,output_1_1) 
psnr_1_2=cv.PSNR(groundTruth,output_1_2) 
psnr_1_3=cv.PSNR(groundTruth,output_1_3) 
psnr_1_4=cv.PSNR(groundTruth,output_1_4) 
psnr_1_5=cv.PSNR(groundTruth,output_1_5)  
psnr_1_6=cv.PSNR(groundTruth,bilateral_OpenCV)  
#Q2
psnr_2_1=cv.PSNR(groundTruth,output_2_1) 
psnr_2_2=cv.PSNR(groundTruth,output_2_2) 
psnr_2_3=cv.PSNR(groundTruth,output_2_3) 
psnr_2_4=cv.PSNR(groundTruth,output_2_4) 
psnr_2_5=cv.PSNR(groundTruth,output_2_5)  
psnr_2_6=cv.PSNR(groundTruth,bilateral_OpenCV_2)  
#Q3
psnr_3_1=cv.PSNR(groundTruth,bilateral_out) 
psnr_3_2=cv.PSNR(groundTruth,bilateral_out_2) 

#TERMINAL OUTPUTS
#Q1
print("Question 1: WINNER WITH THE HIGHEST PSNR(26) IS: aAdaptive mean output is better" )
#Q2
print("Question 2: WINNER WITH THE HIGHEST PSNR(30) IS: 3x3 Gaussian Filter" )
#Q3

print("------------------------------------------")
print("Difference is : ",    np.sum(np.sum(np.abs(bilateral_OpenCV - bilateral_out))))

#OUTPUTS
#Q1
cv.imshow("Q1, 3x3 box filter"           + str(psnr_1_1) , output_1_1)
cv.imshow("Q1, 5x5 box filter"           + str(psnr_1_2) , output_1_2)
cv.imshow("Q1, 3x3 Gaussian filter"      + str(psnr_1_3) , output_1_3)
cv.imshow("Q1, 5x5 Gaussian filter"      + str(psnr_1_4) , output_1_4)
cv.imshow("Q1, my adaptive mean filter"  + str(psnr_1_5) , output_1_5)
cv.imshow("Q1, OpenCV's bilateral filter"+ str(psnr_1_6) , bilateral_OpenCV)
#Q2
cv.imshow("Q2, 3x3 box filter"           + str(psnr_2_1) , output_2_1)
cv.imshow("Q2, 5x5 box filter"           + str(psnr_2_2) , output_2_2)
cv.imshow("Q2, 3x3 Gaussian filter"      + str(psnr_2_3) , output_2_3)
cv.imshow("Q2, 5x5 Gaussian filter"      + str(psnr_2_4) , output_2_4)
cv.imshow("Q2, my adaptive mean filter"  + str(psnr_2_5) , output_2_5)
cv.imshow("Q2, OpenCV's bilateral filter"+ str(psnr_2_6) , bilateral_OpenCV_2)
#Q3
#Differences
cv.imshow("Q3, my adaptive mean filter(5, 3, 0.9))"  + str(psnr_3_1) , bilateral_out)
cv.imshow("Q3, my adaptive mean filter(3, 0.1, 1))"  + str(psnr_3_2) , bilateral_out_2)
cv.imshow('Q3, Difference1' , 100*(abs(bilateral_OpenCV - bilateral_out)))
cv.imshow('Q3, Difference2' , 100*(abs(bilateral_OpenCV - bilateral_out_2)))



cv.waitKey(0)
cv.destroyAllWindows()