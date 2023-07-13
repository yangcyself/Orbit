import matplotlib.pyplot as plt
import numpy as np
import cv2

class InteractiveMask:
    def __init__(self, img):
        self.img = img
        self.mask = np.zeros_like(img[:,:,0], dtype=np.uint8)
        self.points = []

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(img)
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.previous_point = None

    def on_press(self, event):
        if event.inaxes != self.ax: return

        x, y = int(event.xdata), int(event.ydata)
        current_point = (x, y)

        if self.previous_point is not None:
            cv2.line(self.mask, self.previous_point, current_point, 1, thickness=5)
            self.ax.imshow(self.mask, alpha=0.5, cmap='gray')

            if(np.linalg.norm(np.array(self.previous_point) - np.array(current_point))< 1):
                self.on_area_close()
                return

        self.previous_point = current_point
        self.points.append(current_point)
        plt.draw()

    def on_area_close(self):
        if self.points:
            cv2.fillPoly(self.mask, [np.array(self.points)], 1)
            self.ax.imshow(self.mask, alpha=0.5, cmap='gray')
            self.points.clear()
        self.previous_point = None
        plt.draw()

    def get_mask(self):
        return self.mask

# Usage:
img = np.zeros((512, 512, 3), dtype=np.uint8)  # This could be your image
interactor = InteractiveMask(img)
plt.show()

# After closing the plot:
mask = interactor.get_mask()
