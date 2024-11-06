import numpy as np
import cv2
import matplotlib.pyplot as plt

# Configuration parameters for Chan-Vese segmentation
mu = 0.3           # Controls the smoothness of the contour (µ). Increasing it penalizes long contours and keeps the contour closer to the initial region.
nu = 0          # Area penalty (ν). ν>0: A positive value makes the contour tend to minimize its area. ν<0: A negative value encourages the contour to expand.
lambda1 = 1     # Weight for internal areas
lambda2 = 1     # Weight for external areas
epsilon = 1     # Heaviside regularization
time_step = 1e-2  # Adapted time step size
eta = 1e-8      # Curvature regularization
iterations = 1000
tolerance = 0.1  # Convergence tolerance

# Regularized Dirac delta function
def delta_epsilon(phi, epsilon):
    return (epsilon / np.pi) / (epsilon**2 + phi**2)

# Regularized Heaviside function
def heaviside_epsilon(phi, epsilon):
    return 0.5 * (1 + (2 / np.pi) * np.arctan(phi / epsilon))

# Update c1 and c2 (averages inside and outside the contour) for RGB images
def update_c1_c2_colors(image, phi, epsilon):
    H = heaviside_epsilon(phi, epsilon)
    c1 = [np.sum(image[:, :, i] * H) / np.sum(H) for i in range(3)]
    c2 = [np.sum(image[:, :, i] * (1 - H)) / np.sum(1 - H) for i in range(3)]
    return c1, c2

# Update phi using gradient descent for RGB images
def update_phi_colors(phi, image, c1, c2, mu, nu, lambda1, lambda2, epsilon, time_step, eta):
    delta = delta_epsilon(phi, epsilon)
    curvature_term = cv2.Laplacian(phi, cv2.CV_64F)
    
    # Accumulate region force term for each channel
    region_force = sum(-lambda1 * (image[:, :, i] - c1[i])**2 + lambda2 * (image[:, :, i] - c2[i])**2 for i in range(3))
    dphi_dt = delta * (mu * curvature_term - nu + region_force)
    phi += time_step/mu * dphi_dt
    return phi

# Visualize the contour over the RGB image
def plot_contour(image, phi, iteration):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display in matplotlib
    plt.contour(phi, levels=[0], colors='r')  # Draw the contour line where phi=0
    plt.title(f"Iteration: {iteration}")  # Show the current iteration
    plt.axis('off')  # Turn off the axis
    plt.show()

# Chan-Vese implementation for RGB images
def chan_vese_segmentation_colors(image, mu, nu, lambda1, lambda2, epsilon, time_step, iterations, eta, tolerance):
    #image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype('float')
    
    # Normalize the image
    image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
    
    # Get image dimensions
    ni, nj = image.shape[:2]

    # Initialize phi using a circle in the center
    X, Y = np.meshgrid(np.arange(0, nj), np.arange(0, ni), indexing='xy')
    phi = -np.sqrt((X - ni / 2) ** 2 + (Y - nj / 2) ** 2) + 50  # Circle in the center
    
    # Normalize phi to the range [-1, 1]
    phi = 2 * (phi - np.min(phi)) / (np.max(phi) - np.min(phi)) - 1

    for i in range(iterations):
        oldphi=phi.copy()
        c1, c2 = update_c1_c2_colors(image, phi, epsilon)  # Update average colors
        phi = update_phi_colors(phi, image, c1, c2, mu, nu, lambda1, lambda2, epsilon, time_step, eta)  # Update phi
        
        #if i % 20 == 0:  # Visualize every 10 iterations
            #plot_contour(image, phi, i)
        
        # Convergence condition
        phivar = np.sqrt(((phi - oldphi) ** 2).mean())
        if phivar < tolerance:
            print(f"Convergencia alcanzada en {i} iteraciones con cambio promedio: {phivar}")
            break

    mask = heaviside_epsilon(phi, epsilon) > 0.5
    final_mask = (mask * 255).astype(np.uint8)

    return final_mask
    
    """final_mask = heaviside_epsilon(phi, epsilon) > 0.5
    plt.imshow(final_mask, cmap='gray')
    plt.title("Final segmentation result, iteration: "+ str(i))
    plt.axis('off')
    plt.show()

    # Guardar el resultado de la máscara
    mask_filename = 'flores_mask.png'
    cv2.imwrite(mask_filename, (final_mask * 255).astype(np.uint8))
    print(f"Máscara guardada como {mask_filename}")"""



# Run the segmentation
# Execute segmentation based on the type of image
#image_path = 'images/flores.jpg'
#image = cv2.imread(image_path)
#print(image.shape)
#chan_vese_segmentation_colors(image_path, mu, nu, lambda1, lambda2, epsilon, time_step, iterations, eta, tolerance)
