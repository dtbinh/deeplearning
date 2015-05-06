import pyscreenshot as ImageGrab
import PIL
import cv2
import numpy as np
from pymouse import PyMouse

from pymouse import PyMouseEvent


'''
im=ImageGrab.grab(bbox=(65,50,90,100)) # X1,Y1,X2,Y2
print type(im)
im=im.convert('RGB')
print type(im)
im = np.array(im)
print type(im) 

cv_img = im.astype(np.uint8)
cv2.imshow('window', cv_img)
cv2.waitKey(0)
'''

global gX, gY

class ClickOn(PyMouseEvent):
    count = 0;
    def __init__(self):
        PyMouseEvent.__init__(self)
        self.count = 0;

    def click(self, x, y, button, press):
        if button == 1:            
            if press:
                self.count += 1
                gX = x
                gY = y
                print x, y
            
            if self.count == 2:
                self.stop()


if __name__ == "__main__":
    m = PyMouse()
    screenX, screenY = m.screen_size()
 #   C = ClickOn()
 #   C.run()
    count = 0;
    while(True):
        x, y = m.position()
        size = 150
        im = ImageGrab.grab(bbox=(screenX / 2 - size, screenY /2 - size, screenX /2 + size, screenY/2 + size)) # X1,Y1,X2,Y2
      #  im = ImageGrab.grab(bbox=(x - size, y - size, x + size, y + size)) # X1,Y1,X2,Y2
        im = im.convert('RGB')
        im = np.array(im)
        # opencv likes BGR format...
        temp = im[:,:,0].copy()
        im[:,:,0] = im[:,:,2]
        im[:,:,2] = temp

        cv2.imshow('window', im)
        name = str(count)
        name += '.jpg'
        cv2.imwrite(name, cv_img)
        cv2.waitKey(20)
        count += 1;
