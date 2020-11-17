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

def valid(r, c):
    return (r >= 0 and r < BOARD_SZ and c >= 0 and c < BOARD_SZ)


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
    ship_char = ship_chars[x]

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

    ## PREPROCESS PART 2: for EVERY PLACE THAT HAS A SHIP MARKER:
    #  ensure that everything above or to the left within current ship size is OUT
    for row in range(BOARD_SZ):
        for col in range(BOARD_SZ):

            if(board[row][col]!=''): # THERE IS SOMETHING THERE
                for newc in range(col-ship_len+1, col):
                    if(not valid(row, newc)):
                        continue
                    states[row][newc][0]=False # could be overwriting something that's already False, that's fine (can put inside 'if' as well, doesn't matter)
                    remove_ind = get_ind(row, newc, 0)
                    if(remove_ind in rem_states):
                        rem_states.remove(remove_ind)

                for newr in range(row-ship_len+1, row):
                    if(not valid(newr, col)):
                        continue
                    states[newr][col][1]=False
                    remove_ind = get_ind(newr, col, 1)
                    if(remove_ind in rem_states):
                        rem_states.remove(remove_ind)


    rand = random.randint(0, len(rem_states)) # 0 to 199 for 10 x 10
    ind = rem_states[rand]
    i1, i2, i3 = get_parts(ind)

    # (i1, i2) = board position: i1 = row (vert), i2 = col (horiz)
    # i3: 0 = LEFT (horizontal), 1 = TOP (vertical)

    # PLACE THIS SHIP
    for k in range(ship_len):
        if(i3 == 0): # HORIZONTAL placement
            new_i1 = i1
            new_i2 = i2+k

        else:
            new_i1 = i1+k
            new_i2 = i2

        board[new_i1][new_i2]=ship_char

        states[new_i1][new_i2][i3]=False
        states[new_i1][new_i2][(i3+1)%2]=False

        remove_ind = get_ind(new_i1, new_i2, i3)
        if(remove_ind in rem_states):
            rem_states.remove(remove_ind)

        remove_ind = get_ind(new_i1, new_i2, (i3+1)%2)
        if(remove_ind in rem_states):
            rem_states.remove(remove_ind)



print(board)
