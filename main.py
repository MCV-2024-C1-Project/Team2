import cv2
import matplotlib.pyplot as plt


img = cv2.imread('data/BBDD/bbdd_00000.jpg', cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist, color='black')
plt.title('Histograma')
plt.xlabel('Intensidad de píxel')
plt.ylabel('Frecuencia')
plt.xlim([0, 256])  # Asegurarse de que el eje X cubra el rango de 0 a 255
plt.show()

# color
chans = cv2.split(img)
colors = ("b", "g", "r")



#----------<main>-----------

