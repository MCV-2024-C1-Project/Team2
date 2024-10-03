import cv2
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------
# Author: Agustina Ghelfi, Grigor Grigoryan, Philip Zetterberg, Vincent Heuer
# Date: 03.10.2024
# Version: 1.0
# 
# Version 1.1 ads more detailed German descriptions to the notebook
#
# Description:
# This Python script is designed to analyze part lists of wooden crates (HPE 5 tons) in order to calculate manufacturing times.
#
#
# In the future this should be checked with more data from different production locations of Axxum. Furthermore, representation of 
# bottom boards need to be included aswell as a dynamic representation of OSB boards in the sawing process.
# ---------------------------------------------------------------------------------





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

