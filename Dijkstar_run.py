import numpy as np
import time
from queue import PriorityQueue
import cv2
import math
#import imageio as ig
import argparse


class Proj_2:
    def __init__(self, Explored_Node=False, Cost=math.inf, Parent_ind=None):
        self.Explored_Node = Explored_Node
        self.Parent_ind = Parent_ind
        self.Cost = Cost

def Draw_Configuration_Map():
    Map = np.zeros((250, 600, 3))
    cv2.rectangle(Map, (0, 0), (600, 250), (255, 255, 0), 5)
    cv2.rectangle(Map, (100, 0), (150, 100), (255, 0, 0), -1)
    cv2.rectangle(Map, (100, 0), (150, 100), (255, 255, 0), 5)
    cv2.rectangle(Map, (100, 150), (150, 250), (255, 0, 0), -1)
    cv2.rectangle(Map, (100, 150), (150, 250), (255,255 , 0), 5)
    pts = np.array([[300, 50], [225, 82.5], [225, 157.5],
                    [300, 200], [375, 157.5], [375, 82.5]])
    cv2.fillPoly(Map, np.int32([pts]), (255, 0, 0))
    cv2.polylines(Map, np.int32([pts]), True, (255, 255, 0), 5)
    pts = np.array([[460, 25], [460, 225], [510, 125]])
    cv2.fillPoly(Map, [pts], (255,0, 0))
    cv2.polylines(Map, [pts], True, (255, 255, 0), 5)
    cv2.imwrite('Configuration_Map.jpg', Map)
    return Map

def check_input(Map, coordinates):
    w, h = Map.shape[:2]
    x= w-coordinates[1]-1
    y = coordinates[0]
    if not Node_Validity(Map, x, y):
        print('The position is not valid')
        return None
    return x,y


def Node_Validity(Map, x, y):
    w, h = Map.shape[:2]
    
    if (0 <= x < w) and (0 <= y < h) and (Map[x][y] == (0, 0, 0)).all():
        return True
    else:
        return False
def MOVE(Input, MAP, present_node, open_que):
    x_i, y_i = present_node
    
    
   # LEFT
    if Node_Validity(MAP, x_i, y_i-1):
        if (Input[x_i][y_i].Cost + 1) < Input[x_i][y_i-1].Cost:
            Input[x_i][y_i-1].Cost = Input[x_i][y_i].Cost + 1
            Input[x_i][y_i-1].Parent_ind = present_node
            open_que.put((Input[x_i][y_i-1].Cost, (x_i, y_i-1)))

    # RIGHT
    if Node_Validity(MAP, x_i, y_i+1):
        if (Input[x_i][y_i].Cost + 1) < Input[x_i][y_i+1].Cost:
            Input[x_i][y_i+1].Cost = Input[x_i][y_i].Cost + 1
            Input[x_i][y_i+1].Parent_ind = present_node
            open_que.put((Input[x_i][y_i+1].Cost, (x_i, y_i+1)))
    # UP
    if Node_Validity(MAP, x_i-1, y_i):
        # comparing the cost of new child node and current node
        if (Input[x_i][y_i].Cost + 1) < Input[x_i-1][y_i].Cost:
            Input[x_i-1][y_i].Cost = Input[x_i][y_i].Cost + 1
            Input[x_i-1][y_i].Parent_ind = present_node
            open_que.put((Input[x_i-1][y_i].Cost, (x_i-1, y_i)))

    # DOWN
    if Node_Validity(MAP, x_i+1, y_i):
        if (Input[x_i][y_i].Cost + 1) < Input[x_i+1][y_i].Cost:
            Input[x_i+1][y_i].Cost = Input[x_i][y_i].Cost + 1
            Input[x_i+1][y_i].Parent_ind = present_node
            open_que.put((Input[x_i+1][y_i].Cost, (x_i+1, y_i)))

     #DOWN_LEFT
    if Node_Validity(MAP, x_i+1, y_i-1):
        if (Input[x_i][y_i].Cost + 1.4) < Input[x_i+1][y_i-1].Cost:
            Input[x_i+1][y_i-1].Cost = Input[x_i][y_i].Cost + 1.4
            Input[x_i+1][y_i-1].Parent_ind = present_node
            open_que.put(
                (Input[x_i+1][y_i-1].Cost, (x_i+1, y_i-1)))

    # DOWN_RIGHT
    if Node_Validity(MAP, x_i+1, y_i+1):
        if (Input[x_i][y_i].Cost + 1.4) < Input[x_i+1][y_i+1].Cost:
            Input[x_i+1][y_i+1].Cost = Input[x_i][y_i].Cost + 1.4
            Input[x_i+1][y_i+1].Parent_ind = present_node
            open_que.put(
                (Input[x_i+1][y_i+1].Cost, (x_i+1, y_i+1)))

    # UP_LEFT
    if Node_Validity(MAP, x_i-1, y_i-1):
        if (Input[x_i][y_i].Cost + 1.4) < Input[x_i-1][y_i-1].Cost:
            Input[x_i-1][y_i-1].Cost = Input[x_i][y_i].Cost + 1.4
            Input[x_i-1][y_i-1].Parent_ind = present_node
            open_que.put(
                (Input[x_i-1][y_i-1].Cost, (x_i-1, y_i-1)))

    # UP_RIGHT
    if Node_Validity(MAP, x_i-1, y_i+1):
        if (Input[x_i][y_i].Cost + 1.4) < Input[x_i-1][y_i+1].Cost:
            Input[x_i-1][y_i+1].Cost = Input[x_i][y_i].Cost + 1.4
            Input[x_i-1][y_i+1].Parent_ind = present_node
            open_que.put(
                (Input[x_i-1][y_i+1].Cost, (x_i-1, y_i+1)))

    return Input



def backtrack(Input, final_node, copy_map):
    present_node = final_node
    my_queue = []
    backtrack = []
    while present_node is not None:
        my_queue.append(present_node)
        present_node = Input[present_node[0]][present_node[1]].Parent_ind
    my_queue = my_queue[::-1]
    for n in my_queue:
        copy_map[n[0]][n[1]] = (150, 100, 100)
        backtrack.append(np.uint8(copy_map.copy()))
        copy_map[n[0]][n[1]] = (0, 0, 0)
    # cv2.imshow("Frames",backtrack[-1])
    # print(type(backtrack))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('Back_Track.png', backtrack[-1])

def Dijkstra(coordinates):
    
    Map = Draw_Configuration_Map()
    #Initialize parameters
    
    open_que = PriorityQueue()
    w, h = Map.shape[:2]
    Closed_nodes = []
    #First checks the input by passing it through the check_input() function
    start_point = check_input(Map, coordinates['start'])
    end_point = check_input(Map, coordinates['end'])

    if (start_point is None and end_point is None):
        exit(1)

    NODE= [[Proj_2() for i in range(h)] for m in range(w)]
    NODE = np.array(NODE)
    NODE[start_point[0]][start_point[1]].Explored_Node = True
    NODE[start_point[0]][start_point[1]].Cost = 0
   # as per the algorithm we put the starting position into our queue.
    open_que.put((NODE[start_point[0]][start_point[1]].Cost, start_point))
    
    start_time = time.time()
    
    copy_map = Map.copy()
    
    copy_map[end_point[0]][end_point[1]] = (0, 255, 0)
    while open_que:
        present_node= open_que.get()[1]
       
        if present_node == end_point:
          
            end_time = time.time()
            print('time to goal {} sec'.format(end_time-start_time))
            print('Backtracking!, visualisation from goal to start')
            backtrack(NODE, end_point, copy_map)
            break
       
        NODE[present_node[0]][present_node[1]].Explored_Node = True
        NODE = MOVE(NODE, Map, present_node, open_que)
        copy_map[present_node[0]][present_node[1]] = (255, 0, 255)
        Closed_nodes.append(np.uint8(copy_map.copy()))


    cv2.imwrite('Allnodes.png', Closed_nodes[-1])


if __name__ == '__main__':
    #coordinates = getInput()
    inp = argparse.ArgumentParser()
    inp.add_argument('-start', '--start',
                     required=True, nargs='+', type=int)
    inp.add_argument('-end', '--end',
                     required=True, nargs='+', type=int)
    coordinates = vars(inp.parse_args())
    Dijkstra(coordinates)

