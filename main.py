

import cv2
import matplotlib.pyplot as plt

# Local application/library specific imports

import utils

# ---------------------------------------------------------------------------------
# Author: Agustina Ghelfi, Grigor Grigoryan, Philip Zetterberg, Vincent Heuer
# Date: 03.10.2024
# Version: 1.0
# 
# Version 1.0 tackles the tasks of the first week
#
# Description:
# This Python script proposes a solution to building a query for museum paintings using CV techniques such as 1D histogram comparison and more.
#
# In the future this mainscript is expanded to enhance the system's capability and to tackle the upcoming challenges.
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

