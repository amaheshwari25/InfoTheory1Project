## ------- BATTLESHIP -------
## computational solver
## based on probabilistic computations (using randomized sampling of statespace)
##
## OPERATION INSTRUCTIONS:
## - after each command from user, algorithm will suggest which square to probe
## commands from user:
##  - "start" (or anything random) to begin
##  - "miss R C" to indicate that user's guess of [R, C] was a miss
##  - "R C ship_char minR minC maxR maxC" to indicate that user's guess of R C caused a hit that sank ship denoted by ship_char (c, b, d, s, p),
##     which spanned [minR --> maxR] and [minC --> maxC] (note that one of these pairs should always be the same)
##  - "hit R C" to indicate that user's guess of [R, C] was a hit, but did NOT SINK any ships
##
## author: Arya Maheshwari
##
## created: 12.07.20


import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
import copy

### ------------------------------------------------------------ ###
### ---- GLOBAL VARIABLES / HELPER FNs ---- ###

BOARD_SZ = 10

C = 5 # carrier length
B = 4 # battleship length
D = 3 # destroyer length
S = 3 # submarine length
P = 2 # patrol boat length

SHIP_LENS = [P, S, D, B, C]
SHIP_CHARS = ['p', 's', 'd', 'b', 'c']

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

### ------------------------------------------------------------ ###
### ---- COMP GEN STUFF ---- ###

board = np.zeros((BOARD_SZ, BOARD_SZ), dtype='str')


## PROCEDURE: generate random location where to place LEFT or TOP (after determining orientation) ##

def gen_board():
    board = np.zeros((BOARD_SZ, BOARD_SZ), dtype='str')

    start_time = time.time()
    states = np.zeros((BOARD_SZ, BOARD_SZ, 2), dtype='bool')

    rem_states = list()
    ctr = 0
    for elem in states.flatten():
        rem_states.append(ctr)
        ctr+=1


    for x in range(len(SHIP_LENS)): # lowest to greatest length
        ship_len = SHIP_LENS[x]
        ship_char = SHIP_CHARS[x]

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

    end_time = time.time()
    print(end_time - start_time)
    return board

# gen_board()
# gen_board()
# gen_board()
# gen_board()
# gen_board()
# gen_board()
# gen_board()
# gen_board()
# gen_board()
# gen_board()



### ------------------------------------------------------------ ###
### ---- USER FUNCTIONS (actual project) STARTS HERE ---- ###
ACTIVE_HIT = False
hit_loc = [None, None] # [row, col]

LINE_FOUND = False
# adj_line_loc = [None, None]

line_min_loc = [None, None]
line_max_loc = [None, None]


USER_BOARD = np.zeros((BOARD_SZ, BOARD_SZ), dtype='U10')
ship_ind_list = [0, 1, 2, 3, 4]
ship_ind_map = {'c':4, 'b':3, 'd':2, 's':1, 'p':0}

N = 1000 # for simulations
# ---------------- #

# not including usr_brd for now as a parameter just to make this user-end friendly
def hit(r, c):
    global ACTIVE_HIT # why only need this here??
    global LINE_FOUND
    global line_min_loc
    global line_max_loc

    USER_BOARD[r][c]='H'
    if(not ACTIVE_HIT):
        ACTIVE_HIT = True
        hit_loc[0]=r
        hit_loc[1]=c
        line_min_loc = hit_loc
        line_max_loc = hit_loc
    elif (ACTIVE_HIT and not LINE_FOUND):
        LINE_FOUND = True # THIS NEEDS FIXING
        # adj_line_loc[0]=r
        # adj_line_loc[1]=c

        # note that this shouldn't really be needed at this stage (not used in assess_adj), just updated for consistency
        line_min_loc = [min(line_min_loc[0], r), min(line_min_loc[1], c)]
        line_max_loc = [max(line_max_loc[0], r), max(line_max_loc[1], c)]


    else: # both true, in progress of tracking down a hit line
        # just update line bounds
        line_min_loc = [min(line_min_loc[0], r), min(line_min_loc[1], c)]
        line_max_loc = [max(line_max_loc[0], r), max(line_max_loc[1], c)]

def miss(r, c):
    USER_BOARD[r][c]='X'

# note: this modifies the ACTUAL, GLOBAL ship_ind_list (not a copy)
# among other things ... lol
def ship_sunk(ship_char, ship_r_min, ship_c_min, ship_r_max, ship_c_max):
    global ACTIVE_HIT
    global LINE_FOUND
    global line_min_loc
    global line_max_loc

    ship_ind = ship_ind_map[ship_char]
    if(not ship_ind in ship_ind_list):
        print('we have slight problemo')
    else:
        ship_ind_list.remove(ship_ind)

    # change all 'H' for hit on board to the actual ship

    if(ship_r_min == ship_r_max):
        for col in range(ship_c_min, ship_c_max+1):
            USER_BOARD[ship_r_min][col]=ship_char
    else: # ship_c_min == ship_c_max
        for row in range(ship_r_min, ship_r_max+1):
            USER_BOARD[row][ship_c_min]=ship_char

    if (len(ship_ind_list) == 0):
        print("GAME OVER!")
        return 1

    LINE_FOUND = False
    ACTIVE_HIT = False
    hit_loc = [None, None]

    # but check: if there is still a hit remaining not part of this sink, keep active hit on, and set the hit loc (and line min / max loc)
    for row in range(BOARD_SZ):
        for col in range(BOARD_SZ):
            if(USER_BOARD[row][col]=='H'):
                ACTIVE_HIT=True
                print(row, col)
                hit_loc=[row, col]

                min_c, max_c = find_minmax_horiz(USER_BOARD, row, col)
                if(min_c is None):
                    print('looking for vertical lines')
                    min_r, max_r = find_minmax_vert(USER_BOARD, row, col)
                    line_min_loc = [min_r, col]
                    line_max_loc = [max_r, col]
                    if(min_r != max_r):
                        # next find_move will go to looking along this line, rather than re-assessing adjacents (ie finding the other direction)
                        LINE_FOUND = True
                else:
                    line_min_loc = [row, min_c]
                    line_max_loc = [row, max_c]
                    if(min_c != max_c):
                        # next find_move will go to looking along this line, rather than re-assessing adjacents (ie finding the other direction)
                        LINE_FOUND = True



    return 0

def find_minmax_horiz(usr_brd, orig_r, orig_c):
    min_found = False
    min, max = None, None
    for col in range(BOARD_SZ):
        if(usr_brd[orig_r][col]=='H'):
            print('found an H')
            max = col
            if not(min_found):
                min=col

    return min, max

def find_minmax_vert(usr_brd, orig_r, orig_c):
    min_found = False
    min, max = None, None
    for row in range(BOARD_SZ):
        if(usr_brd[row][orig_c]=='H'):
            print('found an H')
            max = row
            if not(min_found):
                min=row

    return min, max

def find_move():
    global ACTIVE_HIT
    global LINE_FOUND

    if(not ACTIVE_HIT):
        return run_sim(USER_BOARD, N, ship_ind_list)

    elif(ACTIVE_HIT and not LINE_FOUND):
        return assess_adj(USER_BOARD, hit_loc, ship_ind_list)

    else: # both are true
        return guess_line(USER_BOARD, line_min_loc, line_max_loc, ship_ind_list)


def play_game(verbose):
    global ACTIVE_HIT
    global LINE_FOUND

    counter = 0
    print("BATTLESHIP: NOTES")
    print("—> 0-indexed: indices are [0, 0] to [9,9]")

    command = input("command: ")
    prev_r = -1
    prev_c = -1
    while(command is not 'stop'):

        comm_list = command.split()
        length = len(comm_list)
        if(length == 3): # standard hit or miss
            # ex: '3 6 hit'
            r = int(comm_list[0])
            c = int(comm_list[1])

            is_hit = (comm_list[2] == 'hit')
            if(is_hit):
                hit(r, c)
            else:
                miss(r, c)

            prev_r = r
            prev_c = c


        if(length == 7): # some ship has been sunk
            # ex: '3 6 b 3 4 3 7'
            r = int(comm_list[0])
            c = int(comm_list[1])
            hit(r, c)

            ship_char = comm_list[2]
            ship_minr = int(comm_list[3])
            ship_minc = int(comm_list[4])
            ship_maxr = int(comm_list[5])
            ship_maxc = int(comm_list[6])
            v = ship_sunk(ship_char, ship_minr, ship_minc, ship_maxr, ship_maxc)
            if(v == 1):
                print("Total moves:", counter)
                return

        print(USER_BOARD)
        if(verbose):
            print('log: ACTIVE HIT?', ACTIVE_HIT)
            print('log: LINE FOUND?', LINE_FOUND)
        print("Suggested move: probe", find_move())

        command = input('command: ')
        counter += 1

    return

# ---------------- #


# generate one possible simulation instance given current board state
def gen_sample(usr_brd_copy, ship_inds):
    global ACTIVE_HIT
    global LINE_FOUND

    '''
    usr_brd_copy: holds information about misses and hits on board - EDITABLE VERSION (i.e. can be used in simulating)
    ship_inds: indices (corresponding to SHIP_LENS / SHIP_CHARS) of ships still in play (0 = p --> 4 = c)
    '''

    # start_time = time.time()

    states = np.zeros((BOARD_SZ, BOARD_SZ, 2), dtype='bool')

    rem_states = list()
    ctr = 0
    for elem in states.flatten():
        rem_states.append(ctr)
        ctr+=1



    for x in range(len(SHIP_LENS)): # lowest to greatest length
        if(x not in ship_inds):
            continue
        # print(x)
        ship_len = SHIP_LENS[x]
        ship_char = SHIP_CHARS[x]

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

                if(usr_brd_copy[row][col]!=''): # THERE IS SOMETHING THERE
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


        rand = random.randint(0, len(rem_states)-1) # 0 to 199 for 10 x 10
        # print(len(rem_states))
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

            usr_brd_copy[new_i1][new_i2]='SIM_'+ship_char

            states[new_i1][new_i2][i3]=False
            states[new_i1][new_i2][(i3+1)%2]=False

            remove_ind = get_ind(new_i1, new_i2, i3)
            if(remove_ind in rem_states):
                rem_states.remove(remove_ind)

            remove_ind = get_ind(new_i1, new_i2, (i3+1)%2)
            if(remove_ind in rem_states):
                rem_states.remove(remove_ind)

    # end_time = time.time()
    # print(end_time - start_time)
    # print('done')
    return usr_brd_copy

# run_sim is basically the function you want to call if not in an active hit situation
def run_sim(usr_brd, n, ship_inds):
    global ACTIVE_HIT
    global LINE_FOUND

    '''
    usr_brd: holds (actual, not simulated) information about misses and hits on board
    n: number of simulations to run
    ship_inds: indices (corresponding to SHIP_LENS / SHIP_CHARS) of ships still in play (0 = p --> 4 = c)
    '''

    master_counts = np.zeros((BOARD_SZ, BOARD_SZ), dtype='int')

    max_count = 0
    max_count_r = -1
    max_count_c = -1

    for i in range(n): # number of simulations

        # create a deep copy of usr_brd array
        usr_copy = copy.deepcopy(usr_brd)

        # generate simulation board
        sample = gen_sample(usr_copy, ship_inds)
        # if(i == 0):
        #     print(sample)

        # get counts
        for row in range(BOARD_SZ):
            for col in range(BOARD_SZ):

                if(len(sample[row][col]) > 1): # only true if something was placed here in the simulation (then it'll be a string of length 5 ("SIM_X"))
                    master_counts[row][col]+=1

                if(master_counts[row][col] > max_count): # keep a running max
                    max_count = master_counts[row][col]
                    max_count_r = row
                    max_count_c = col


    # after all simulations, return the location to guess!
    return max_count_r, max_count_c

# what to call if ACTIVE_HIT but *not* LINE_FOUND
def assess_adj(usr_brd, hit_loc, ship_inds):
    global ACTIVE_HIT
    global LINE_FOUND

    hit_r = hit_loc[0]
    hit_c = hit_loc[1]

    # keep a running max of which adjacent spot (L, R, U, D) to pick
    max_configs = 0
    max_conf_ind = -1

    ADJ_CTR = [0, 0, 0, 0] # order: L, R, U, D (left, right, up, down counts)
    # TBD: unclear if there is some nonuniform prob dist here between the different possibilities? (like [literal] edge cases less likely to be generated?)
    # FOR NOW: assuming "IF POSSIBLE", then "EQUAL CHANCE OF FORM"
    # ^^ this assumption is GLOBALLY TRUE: assuming all possible ship orientations are EQUALLY LIKELY (is that valid?? depends on ship assignment, i guess...)

    for x in range(len(SHIP_LENS)): # lowest to greatest length
        if(x not in ship_inds): # if it's already been found, ignore
            continue
        ship_len = SHIP_LENS[x]
        ship_char = SHIP_CHARS[x]

        # try horizontal orientations
        for left_ind in range(hit_c-ship_len+1, hit_c+1): # the left index of ship could be anywhere between these two values (pre-additional checking)
            VIABLE = True
            for step in range(left_ind, left_ind+ship_len):
                # CHECK THIS UPDATE: MADE THIS 'H' condition stronger, right? (it HAS to be the hit location too, because we don't want to re-guess earlier H things)
                if not(valid(hit_r, step) and (usr_brd[hit_r][step]=='' or (usr_brd[hit_r][step]=='H' and step==hit_c))): # other placed ships will have lowercase letter, these are not valid; misses are X; only CURRENT HITS are 'H'
                    VIABLE=False

            # print(VIABLE)
            # print(usr_brd[hit_r][hit_c-1])

            if(VIABLE):
                if(left_ind != hit_c):
                    ADJ_CTR[0]+=1 # one config includes LEFT spot
                    if(ADJ_CTR[0]>max_configs):
                        max_configs = ADJ_CTR[0]
                        max_conf_ind=0

                if(left_ind != hit_c-ship_len+1):
                    ADJ_CTR[1]+=1 # one config includes RIGHT spot
                    if(ADJ_CTR[1]>max_configs):
                        max_configs = ADJ_CTR[1]
                        max_conf_ind=1


        # continue to vertical orientations
        # (note that "up" = top, "down" = bottom really: just using same words as above for consistency)
        for up_ind in range(hit_r-ship_len+1, hit_r+1): # the left index of ship could be anywhere between these two values (pre-additional checking)
            VIABLE = True
            for step in range(up_ind, up_ind+ship_len):
                if not(valid(step, hit_c) and (usr_brd[step][hit_c]=='' or (usr_brd[step][hit_c]=='H' and step==hit_r))): # other placed ships will have lowercase letter, these are not valid; misses are X
                    VIABLE=False

            if(VIABLE):
                if(up_ind != hit_r):
                    ADJ_CTR[2]+=1 # one config includes UP spot
                    if(ADJ_CTR[2]>max_configs):
                        max_configs = ADJ_CTR[2]
                        max_conf_ind=2

                if(up_ind != hit_r-ship_len+1):
                    ADJ_CTR[3]+=1 # one config includes DOWN spot
                    if(ADJ_CTR[3]>max_configs):
                        max_configs = ADJ_CTR[3]
                        max_conf_ind=3

    if(max_conf_ind == -1):
        print('we have slight problemo')
        return

    if(max_conf_ind == 0):
        return hit_r, hit_c-1
    elif(max_conf_ind == 1):
        return hit_r, hit_c+1
    elif(max_conf_ind == 2):
        return hit_r-1, hit_c
    else:
        return hit_r+1, hit_c

# call when BOTH ACTIVE_HIT and LINE_FOUND (or equivalently just LINE_FOUND, since ACTIVE_HIT is basically a precondition)
# generally will just pass in the global variables for min_loc and max_loc
def guess_line(usr_brd, min_loc, max_loc, ship_inds):
    global ACTIVE_HIT
    global LINE_FOUND

    line_len = max(max_loc[0]-min_loc[0], max_loc[1]-min_loc[1]) # one of these values should be 0, the other should be what we want (and positive)

    if(max_loc[0] == min_loc[0]): # rows are same, so horizontal
        return guess_horiz_line(usr_brd, min_loc, max_loc, line_len, ship_inds)
    else:
        return guess_vert_line(usr_brd, min_loc, max_loc, line_len, ship_inds)

# HELPERS
def guess_vert_line(usr_brd, min_loc, max_loc, line_len, ship_inds):
    global ACTIVE_HIT
    global LINE_FOUND

    '''
    min_loc and max_loc will be (a, c) --> (b, c), where b > a
    '''

    c = min_loc[1]
    up_loc = (min_loc[0]-1, c)
    down_loc = (max_loc[0]+1, c)

    max_configs = 0

    up_cts = 0
    down_cts = 0

    up_max = True

    for x in range(len(SHIP_LENS)): # lowest to greatest length
        if(x not in ship_inds): # if it's already been found, ignore
            continue
        ship_len = SHIP_LENS[x]
        ship_char = SHIP_CHARS[x]

        for up_ind in range(up_loc[0]-ship_len+1, down_loc[0]): # the left index of ship could be anywhere between these two values (pre-additional checking)
            VIABLE = True
            for step in range(up_ind, up_ind+ship_len):
                # here, unlike assess_adj, we are chill with line hits that aren't the starting/original hit position
                if not(valid(step, c) and (usr_brd[step][c]=='' or usr_brd[step][c]=='H')): # other placed ships will have lowercase letter, these are not valid; misses are X; only CURRENT HITS are 'H'
                    VIABLE=False

            if(VIABLE):
                # CHECK THESE CONDITIONS....
                if(up_ind <= up_loc[0]):
                    up_cts+=1 # one config includes UP spot
                    if(up_cts>max_configs):
                        max_configs = up_cts
                        up_max=True

                if(up_ind+ship_len-1 >= down_loc[0]):
                    down_cts+=1 # one config includes DOWN spot
                    if(down_cts>max_configs):
                        max_configs = down_cts
                        up_max=False

    if(up_cts + down_cts == 0): # if they're both still at 0
        # WE NEED TO FLIP DIRECTION! (i.e. we can't place any ships along this line anymore...)
        LINE_FOUND = False
        assess_adj(usr_brd, hit_loc, ship_inds) # right? just rotate around original hit if still active; if that doesn't work

    elif(up_cts > down_cts):
        return up_loc[0], up_loc[1]
    else:
        return down_loc[0], down_loc[1]

def guess_horiz_line(usr_brd, min_loc, max_loc, line_len, ship_inds):
    global ACTIVE_HIT
    global LINE_FOUND


    '''
    min_loc and max_loc will be (r, a) --> (r, b), where b > a
    '''

    r = min_loc[0]
    left_loc = (r, min_loc[1]-1)
    right_loc = (r, max_loc[1]+1)

    left_cts = 0
    right_cts = 0

    max_configs = 0

    left_max = True

    for x in range(len(SHIP_LENS)): # lowest to greatest length
        if(x not in ship_inds): # if it's already been found, ignore
            continue
        ship_len = SHIP_LENS[x]
        ship_char = SHIP_CHARS[x]

        for left_ind in range(left_loc[1]-ship_len+1, right_loc[1]): # the left index of ship could be anywhere between these two values (pre-additional checking)
            VIABLE = True
            for step in range(left_ind, left_ind+ship_len):
                # here, unlike assess_adj, we are chill with line hits that aren't the starting/original hit position
                if not(valid(r, step) and (usr_brd[r][step]=='' or usr_brd[r][step]=='H')): # other placed ships will have lowercase letter, these are not valid; misses are X; only CURRENT HITS are 'H'
                    VIABLE=False

            if(VIABLE):
                # CHECK THESE CONDITIONS....
                if(left_ind <= left_loc[1]):
                    left_cts+=1 # one config includes UP spot
                    if(left_cts>max_configs):
                        max_configs = left_cts
                        left_max=True

                if(left_ind+ship_len-1 >= right_loc[1]):
                    right_cts+=1 # one config includes DOWN spot
                    if(right_cts>max_configs):
                        max_configs = right_cts
                        left_max=False

    if(left_cts + right_cts == 0): # if they're both still at 0
        # WE NEED TO FLIP DIRECTION! (i.e. we can't place any ships along this line anymore...)
        LINE_FOUND = False
        assess_adj(usr_brd, hit_loc, ship_inds) # right? just rotate around original hit if still active; if that doesn't work

    elif(left_cts > right_cts):
        return left_loc[0], left_loc[1]
    else:
        return right_loc[0], right_loc[1]




# ----- PLAY THE GAME! ------ #
play_game(verbose=False)










### -------------------------------------------------------------------- ###
### ---- CURRENTLY UNUSED STUFF ---- ###

# # note: this is a little extra considering the process could be iterative, but restarting brute force each turn is fine for this:
# #  only ~1000 operations
# #
# def assess_val(usr_brd):
#     '''
#      usr_brd: holds information about misses and hits on board
#     '''
#
#     dp = np.zeros((BOARD_SZ, BOARD_SZ, len(SHIP_CHARS)+1, 2))
#
#     for length in range(1, len(SHIP_CHARS)+1):
#         for dir in range(0, 2):
#             for row in range(BOARD_SZ): # note: going from 0 to LEN-1, so DP is for whether RIGHT or BOTTOM edges ok
#                 for col in range(BOARD_SZ):
#
#                     valid=(usr_brd[row][col]!='X')
#                     if(length > 1):
#                         if(dir == 0): # horizontal
#                             valid=(valid and valid(row, col-1) and dp[row][col-1][length-1][dir])
#                         else:         # vertical
#                             valid=(valid and valid(row-1, col) and dp[row-1][col][length-1][dir])
#
#                     dp[row][col][length][dir]=valid
#
#     return dp
#
# def probe(i, j, usr_brd, comp_board):
#     if(comp_board[i][j]==''):
#         usr_brd[i][j]='X'
#         print(i, j, "miss")
#     else:
#         usr_brd[i][j]='H'
#         print(i, j, "hit")


# prob_board = np.zeros((BOARD_SZ, BOARD_SZ))
#
#
# for row in range(BOARD_SZ):
#     for col in range(BOARD_SZ):
#
#         if(board[row][col]!=''): # THERE IS SOMETHING THERE
#             for newc in range(col-ship_len+1, col):
#                 if(not valid(row, newc)):
#                     continue
#                 states[row][newc][0]=False # could be overwriting something that's already False, that's fine (can put inside 'if' as well, doesn't matter)
#                 remove_ind = get_ind(row, newc, 0)
#                 if(remove_ind in rem_states):
#                     rem_states.remove(remove_ind)
#
#             for newr in range(row-ship_len+1, row):
#                 if(not valid(newr, col)):
#                     continue
#                 states[newr][col][1]=False
#                 remove_ind = get_ind(newr, col, 1)
#                 if(remove_ind in rem_states):
#                     rem_states.remove(remove_ind)
