import numpy as np
import cv2

def Draw_Config_Map():
    #Created a black image of shape same as the given map
    Mask=np.zeros((250,600,3))
    #Now we will draw shapes using cv2 functions
    #Dimensions of the shapes are original+5mm for clearance
    Rectangle_1=[(100,0),(100,100),(150,100),(150,0)]
    cv2.fillPoly(Mask,np.array([Rectangle_1]),(255,0,0))
    Rectangle_2=[(100,150),(100,250),(150,250),(150,150)]
    cv2.fillPoly(Mask,np.array([Rectangle_2]),(255,0,0))
    Traingle=[(460,25),(460,225),(510,125)]
    cv2.fillPoly(Mask,np.array([Traingle]),(255,0,0))
    Hexagon=[(235.048,87.5),(300,50),(364.952,87.5),(364.952,162.5),(300,200),(235.048,162.5)]
    cv2.fillPoly(Mask,np.int32([Hexagon]),(255,0,0))
    cv2.imshow("Rectangle",Mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


Draw_Config_Map()