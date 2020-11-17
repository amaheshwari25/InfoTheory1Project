import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import random

BOARD_SZ = 10

# index = i1 + BOARD_SZ*i2 + (BOARD_SZ^2)*i3
def get_ind(i1, i2, i3):
    return i1 + BOARD_SZ*i2 + (BOARD_SZ**2)*i3

def get_parts(ind):
    i1 = ind % BOARD_SZ
    i2 = int(ind / BOARD_SZ) % BOARD_SZ
    i3 = int(ind / BOARD_SZ**2) % 2
    return i1, i2, i3


board = np.zeros((BOARD_SZ, BOARD_SZ), dtype='str')

## PROCEDURE: generate random location where to place LEFT or TOP (after determining orientation) ##

c = 5 # carrier length
b = 4 # battleship length
d = 3 # destroyer length
s = 3 # submarine length
p = 2 # patrol boat length

ship_lens = [p, s, d, b, c]
ship_chars = ['p', 's', 'd', 'b', 'c']

states = np.zeros((BOARD_SZ, BOARD_SZ, 2), dtype='bool')

rem_states = list()
ctr = 0
for elem in states.flatten():
    rem_states.append(ctr)
    ctr+=1


for x in range(len(ship_lens)): # lowest to greatest length
    ship_len = ship_lens[x]
    ## PREPROCESS:  REMOVE EDGE THINGS: anything either RIGHT or DOWN of (BOARD_SZ - ship_len) ##

    # dir = 0: horizontal (so COLs clipped); dir = 1: vertical (so ROWs clipped)
    for dim1 in range(BOARD_SZ - ship_len + 1, BOARD_SZ):
        for dim2 in range(BOARD_SZ):
            states[dim1][dim2][1]=False
            states[dim2][dim1][0]=False
            remove_ind_vert = get_ind(dim1, dim2, 1)
            remove_ind_horiz = get_ind(dim2, dim1, 0)
            if(remove_ind_vert in rem_states):
                rem_states.remove(remove_ind_vert)
            if(remove_ind_horiz in rem_states):
                rem_states.remove(remove_ind_horiz)

    # print(ship_len)
    # print(len(rem_states))


    rand = random.randint(0, len(rem_states)) # 0 to 199 for 10 x 10
    ind = rem_states[rand]
    i1, i2, i3 = get_parts(ind)

    # (i1, i2) = board position: i1 = row (vert), i2 = col (horiz)
    # i3: 0 = LEFT (horizontal), 1 = TOP (vertical)

    # PLACE THIS SHIP
    for k in range(len(ship_len)):
        if(i3 == 0): # HORIZONTAL placement
            states[i1][i2+k][i3]=False
            remove_ind = get_ind(i1, i2+k, i3)
            if(remove_ind in rem_states):
                rem_states.remove(remove_ind)
        else:
            states[i1+k][i2][i3]=False
            remove_ind = get_ind(i1+k, i2, i3)
            if(remove_ind in rem_states):
                rem_states.remove(remove_ind)


    # orient = random.randint(0, 1) # 0 = horizontal, 1 = vertical
    # dim1_edge_loc = random.randint(0, BOARD_SZ-ship_len) # the index on the constrained dimension
    # dim2_edge_loc = random.randint(0, BOARD_SZ)
    #
    # if(orient == 0): #horizontal
    #     for i in range()
    # else:
    #     asdf
