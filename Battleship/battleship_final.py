## ------- BATTLESHIP solver code -------
##
## author: Arya Maheshwari
##         with Aditya Singhvi
##
## written for project in Advanced Topics in Math: Information Theory 1
##
## created: 01.09.21 / 12.07.20


import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
import copy
from sortedcontainers import SortedList, SortedDict, SortedSet

### ------------------------------------------------------------ ###
### ------------------- GLOBAL VARIABLES ------------------- ###

BOARD_SZ = 10

C = 5 # carrier length
B = 4 # battleship length
D = 3 # destroyer length
S = 3 # submarine length
P = 2 # patrol boat length

SHIP_LENS = [P, S, D, B, C]
SHIP_CHARS = ['p', 's', 'd', 'b', 'c']

GAME_OVER = False

ACTIVE_HIT = False
hit_loc = [None, None] # [row, col]

LINE_FOUND = False
# adj_line_loc = [None, None]

line_min_loc = [None, None]
line_max_loc = [None, None]


USER_BOARD = np.zeros((BOARD_SZ, BOARD_SZ), dtype=int)

ship_ind_list = [0, 1, 2, 3, 4]
ship_ind_map = {'c':4, 'b':3, 'd':2, 's':1, 'p':0}

N = 1000 # for simulations

HORIZONTAL = np.zeros((len(SHIP_CHARS)), dtype='bool') # true if ship oriented horizontally, false otherwise (p, s, d, b, c)

COMPUTER_BOARD = None

# ----------------- MINI-HELPER METHODS -------------------- #

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

def grid_ind(r, c):
    global BOARD_SZ
    return BOARD_SZ*r + c

# ---------------- #
# ALL OF THESE MUST BE INTEGERS
TOP_THRESH = int(max(20, 0.0002 * N)) # 20 boards each time, after 0
N_SAMPLES = 10000
MAX_THRESH = min(50000, N_SAMPLES)

SAMPLE_ARRAY = np.zeros((N_SAMPLES, BOARD_SZ, BOARD_SZ), dtype='int')

SORTED_INDS = SortedList()
IND_VALUES = {}
for i in range(N_SAMPLES):
    SORTED_INDS.add((0, i))
    IND_VALUES[i]=0


for r in range(BOARD_SZ):
    for c in range(BOARD_SZ):
        USER_BOARD[r][c]=-1 # to start off

GUESSED_SET = set() #0 to 99, 10*r + c to note if (r, c) guessed or not


SINK_CTR = 0
END_FLAG = False
VERBOSE = False

# data structures: used to store queries, sinks, and ship placement
query_log = np.zeros((BOARD_SZ**2, 3), dtype=int) # [0] = r, [1] = c, [2] = response_val
sink_data = np.zeros((len(SHIP_LENS), 4), dtype=int) #[0] = sink_r, [1] = sink_c, [2] = dr, [3] = dc
fixed_ship = np.zeros((len(SHIP_LENS), 5), dtype=int) # [0] = sink_r, [1] = sink_c, [2] = other_r, [3] = other_c, [4] = len
fs_ctr = 0

### ------------------------------------------------------------ ###
### -------------------- INTERNAL GAMEPLAY FUNCTIONS -------------------- ###

# what to call if ACTIVE_HIT but *not* LINE_FOUND
def assess_adj(usr_brd, hit_loc):
    global ACTIVE_HIT
    global LINE_FOUND

    hit_r = hit_loc[0]
    hit_c = hit_loc[1]

    # keep a running max of which adjacent spot (L, R, U, D) to pick
    max_configs = 0
    max_conf_ind = -1

    ADJ_CTR = [0, 0, 0, 0] # order: L, R, U, D (left, right, up, down counts)

    for x in range(len(SHIP_LENS)): # lowest to greatest length
        ship_len = SHIP_LENS[x]
        ship_char = SHIP_CHARS[x]

        # try horizontal orientations
        for left_ind in range(hit_c-ship_len+1, hit_c+1): # the left index of ship could be anywhere between these two values (pre-additional checking)
            VIABLE = True
            for step in range(left_ind, left_ind+ship_len):

                # TBD: WHAT EXACTLY ARE WE ALLOWING IN THIS METHOD?
                if ((not valid(hit_r, step)) or (usr_brd[hit_r][step]>=0 and step!=hit_c)): # saying a HIT=1= UNVIABLE, so that it doesn't reguess? but is this dumb?
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

                # TBD: WHAT EXACTLY ARE WE ALLOWING IN THIS METHOD?
                if (not valid(step, hit_c) or (usr_brd[step][hit_c]>=0 and step!=hit_r)): # saying a HIT =1= UNVIABLE, so that it doesn't reguess? but is this dumb?
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
        print(USER_BOARD)
        print(hit_loc)
        print(ADJ_CTR)
        print('Defaulting to random guess...')

        ACTIVE_HIT = False # turn this off so we don't loop into this mode
        return default_guess(usr_brd)

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
def guess_line(usr_brd, hit_loc, min_loc, max_loc):
    global ACTIVE_HIT
    global LINE_FOUND

    line_len = max(max_loc[0]-min_loc[0], max_loc[1]-min_loc[1]) # one of these values should be 0, the other should be what we want (and positive)

    if(max_loc[0] == min_loc[0]): # rows are same, so horizontal
        return guess_horiz_line(usr_brd, hit_loc, line_len)
    else:
        return guess_vert_line(usr_brd, hit_loc, line_len)
# HELPERS
def guess_vert_line(usr_brd, hit_loc, line_len):
    global ACTIVE_HIT
    global LINE_FOUND
    global line_min_loc
    global line_max_loc

    '''
    min_loc and max_loc will be (a, c) --> (b, c), where b > a
    '''

    c = hit_loc[1]
    min_loc_r, max_loc_r  = find_minmax_vert(usr_brd, hit_loc[0], hit_loc[1])

    up_loc = (min_loc_r-1, c)
    down_loc = (max_loc_r+1, c)

    max_configs = 0

    up_cts = 0
    down_cts = 0

    up_max = True

    for x in range(len(SHIP_LENS)): # lowest to greatest length
        ship_len = SHIP_LENS[x]
        ship_char = SHIP_CHARS[x]

        for up_ind in range(up_loc[0]-ship_len+1, down_loc[0]): # the left index of ship could be anywhere between these two values (pre-additional checking)
            VIABLE = True
            for step in range(up_ind, up_ind+ship_len):
                # here, unlike assess_adj, we are chill with line hits that aren't the starting/original hit position
                if(not valid(step, c) or usr_brd[step][c]==0): # other placed ships will have lowercase letter, these are not valid; misses are X; only CURRENT HITS are 'H'
                    VIABLE=False

            if(VIABLE):
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

        # new, rare edge case: there might still be a horiz line active..
        # minc, maxc = find_minmax_horiz(usr_brd, hit_loc[0], hit_loc[1])
        # if(minc != maxc):
        #     line_min_loc = [hit_loc[0], minc]
        #     line_max_loc = [hit_loc[0], maxc]
        #     return guess_horiz_line(usr_brd, line_min_loc, line_max_loc, maxc-minc+1, ship_inds)

        LINE_FOUND = False
        line_min_loc = hit_loc # NEW
        line_max_loc = hit_loc # NEW

        # this is its all-catcher...which should trigger assess_adj's default catcher as needed?
        return assess_adj(usr_brd, hit_loc) # right? just rotate around original hit if still active; if that doesn't work

    elif(up_cts > down_cts):
        return up_loc[0], up_loc[1]
    else:
        return down_loc[0], down_loc[1]
# direct analog to guess_vert_line
def guess_horiz_line(usr_brd, hit_loc, line_len):
    global ACTIVE_HIT
    global LINE_FOUND
    global line_min_loc
    global line_max_loc


    '''
    min_loc and max_loc will be (r, a) --> (r, b), where b > a
    '''

    r = hit_loc[0]
    min_loc_c, max_loc_c  = find_minmax_horiz(usr_brd, hit_loc[0], hit_loc[1])

    left_loc = (r, min_loc_c-1)
    right_loc = (r, max_loc_c+1)

    left_cts = 0
    right_cts = 0

    max_configs = 0

    left_max = True

    for x in range(len(SHIP_LENS)): # lowest to greatest length
        ship_len = SHIP_LENS[x]
        ship_char = SHIP_CHARS[x]

        for left_ind in range(left_loc[1]-ship_len+1, right_loc[1]): # the left index of ship could be anywhere between these two values (pre-additional checking)
            VIABLE = True
            for step in range(left_ind, left_ind+ship_len):
                # here, unlike assess_adj, we are chill with line hits that aren't the starting/original hit position
                if(not valid(r, step) or usr_brd[r][step]==0):
                    VIABLE=False

            if(VIABLE):
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

        # # new, rare edge case: there might still be a vertical line active..
        # minr, maxr = find_minmax_vert(usr_brd, hit_loc[0], hit_loc[1])
        # if(minr != maxr):
        #     line_min_loc = [minr, hit_loc[1]]
        #     line_max_loc = [maxr, hit_loc[1]]
        #     return guess_vert_line(usr_brd, line_min_loc, line_max_loc, maxr-minr+1, ship_inds)

        LINE_FOUND = False
        line_min_loc = hit_loc # NEW
        line_max_loc = hit_loc # NEW

        # this is its all-catcher...which should trigger assess_adj's default catcher as needed?
        return assess_adj(usr_brd, hit_loc) # right? just rotate around original hit if still active; if that doesn't work

    elif(left_cts > right_cts):
        return left_loc[0], left_loc[1]
    else:
        return right_loc[0], right_loc[1]

# fan out from (orig_r, orig_c) in horizontal direction and find bounds
def find_minmax_horiz(usr_brd, orig_r, orig_c):
    print("find_minmax_horiz called on 0-ind position", orig_r, orig_c)
    #precondition: usr_brd[orig_r][orig_c]='H'
    if(usr_brd[orig_r][orig_c]<1):
        print("find_minmax_horiz called incorrectly: 0-ind position", orig_r, orig_c, "is not a hit")
        return None

    min_col = orig_c
    while(valid(orig_r, min_col) and usr_brd[orig_r][min_col]>=1):
        min_col-=1

    max_col = orig_c
    while(valid(orig_r, max_col) and usr_brd[orig_r][max_col]>=1):
        max_col+=1

    return min_col+1, max_col-1 # shift one back, because went one extra too far

# fan out from (orig_r, orig_c) in vertical direction and find bounds
def find_minmax_vert(usr_brd, orig_r, orig_c):
    print("find_minmax_vert called on 0-ind position", orig_r, orig_c)

    #precondition: usr_brd[orig_r][orig_c]='H'
    if(usr_brd[orig_r][orig_c]<1):
        print("find_minmax_horiz called incorrectly: 0-ind position", orig_r, orig_c, "is not a hit")
        return None

    min_row = orig_r
    while(valid(min_row, orig_c) and usr_brd[min_row][orig_c]>=1):
        min_row-=1

    max_row = orig_r
    while(valid(max_row, orig_c) and usr_brd[max_row][orig_c]>=1):
        max_row+=1

    return min_row+1, max_row-1 # shift one back, because went one extra too far


# default, if other guessing fails: tries to find spots where a 2-ship could be placed (in any direction) and randomly guesses
def default_guess(usr_brd):
    poss_guesses = {1:[], 2:[], 3:[], 4:[]}
    for row in range(BOARD_SZ):
        for col in range(BOARD_SZ):
            # search for whether (row, col) can be LEFT, RIGHT, UP, or DOWN - and add for each
            unblckd = unblocked(usr_brd, row, col)
            if(not unblckd):
                continue

            dr=[0, 0, -1, 1]
            dc=[-1, 1, 0, 0]
            orient = 0
            for i in range(len(dr)):
                if(unblocked(usr_brd, row+dr, col+dc)):
                    orient+=1

            poss_guesses[orient].append(10*row+col)

    for i in range(len(dr)):
        guess_list = poss_guesses[len(dr)-i-1]
        if(len(guess_list)>0):
            guess_move = guess_list[random.randint(0, len(guess_list)-1)]
            return int(guess_move/10), int(guess_move%10)

    return -1, -1

# check that board[r][c] is valid and non-zero
def unblocked(usr_brd, r, c):
    return valid(r, c) and (usr_brd[r][c]!=0)

# generate one possible simulation instance given current board state
def gen_sample(ship_inds, array_ind, usr_brd):

    global BOARD_SZ
    global SHIP_LENS
    global SHIP_CHARS
    global SAMPLE_ARRAY

    '''
    usr_brd_copy: holds information about misses and hits on board - EDITABLE VERSION (i.e. can be used in simulating)
    ship_inds: indices (corresponding to SHIP_LENS / SHIP_CHARS) of ships still in play (0 = p --> 4 = c)
    '''

    # start_time = time.time()

    new_board = np.zeros((BOARD_SZ, BOARD_SZ), dtype=int)
    for r in range(BOARD_SZ):
        for c in range(BOARD_SZ):
            if(usr_brd[r][c]==3):
                new_board[r][c]=3
            # OLD PROCEDURE: propagated misses
            # if(usr_brd[r][c]==0):
            #     new_board[r][c]=-1 # WATCH OUT: sample board conventions are DIFFERENT from user_board: a 0 on user_board = miss = -1 on sample board
            # elif(usr_brd[r][c]==3):
            #     new_board[r][c]=3 # here, sample board 3 = user board 3 = confirmed ship
            #     # note that ANYWHERE a 3 gets placed should NEVER be queried again: already guessed, that's how we know it's a ship


    # rem_states = list()
    rem_states = set()
    for elem in range(BOARD_SZ*BOARD_SZ*2):
        # rem_states.append(ctr)
        rem_states.add(elem)

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

                # new_board instead of direct SAMPLE_ARRAY is NEW
                if(new_board[row][col]!=0): # THERE IS SOMETHING THERE: 1 and 2 for other ships placed, (DEPR: -1 for miss), 3 for a confirmed ship placed
                    for newc in range(col-ship_len+1, col+1): # NEW ADDITION: REMEMBER — *ALSO GET RID OF THIS POSITION*, and range is < not ≤
                        if(not valid(row, newc)):
                            continue
                        remove_ind = get_ind(row, newc, 0)
                        if(remove_ind in rem_states):
                            rem_states.remove(remove_ind)

                    for newr in range(row-ship_len+1, row+1): # NEW ADDITION: REMEMBER — *ALSO GET RID OF THIS POSITION*, and range is < not ≤
                        if(not valid(newr, col)):
                            continue
                        remove_ind = get_ind(newr, col, 1)
                        if(remove_ind in rem_states):
                            rem_states.remove(remove_ind)


        ind = random.sample(rem_states, 1)[0]
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

            if(k == 0 or k==ship_len-1):
                new_board[new_i1][new_i2]=2
            else:
                new_board[new_i1][new_i2]=1 # 1 = mid-ship

            remove_ind = get_ind(new_i1, new_i2, i3)
            if(remove_ind in rem_states):
                rem_states.remove(remove_ind)

            remove_ind = get_ind(new_i1, new_i2, (i3+1)%2)
            if(remove_ind in rem_states):
                rem_states.remove(remove_ind)

    SAMPLE_ARRAY[array_ind]=new_board
    # print(SAMPLE_ARRAY[array_ind])
    return

# calls gen_sample n times, and sends in the appropriate list of ships that need placing
def gen_n_samples(n, usr_brd):
    global USER_BOARD
    global ship_ind_list
    global fs_ctr
    global SHIP_LEN

    if(fs_ctr==0): # this is the initial sample, then
        print("Generating initial sample of configurations...this could take some time.")

    start = time.time()

    updated_shipind_list = copy.deepcopy(ship_ind_list)
    for x in range(fs_ctr):
        x_len = fixed_ship[x][4]
        removable = -1
        for val in updated_shipind_list:
            if(SHIP_LENS[val]==x_len):
                removable=val
                continue
        if(removable>-1):
            updated_shipind_list.remove(removable)
        else:
            print("should NEVER have an unremovable length / length that is not found in ship_ind_list...")

    for i in range(n):
        gen_sample(updated_shipind_list, i, usr_brd)
        # gen_sample(ship_ind_list, i, usr_brd)
        # if(i%(n/20)==0):
        #     print(i)

    end = time.time()
    if(fs_ctr==0 and VERBOSE):
        print("gen sample total time:", end-start)

    return

# check across board to see if we can determine any ship's length for sure
def evaluate_fixed_ship(usr_brd):
    global SINK_CTR
    global sink_data
    global fs_ctr
    global fixed_ship


    new_fs_ctr = 0
    for i in range(SINK_CTR):
        sink_r, sink_c, dr, dc = sink_data[i][0], sink_data[i][1], sink_data[i][2], sink_data[i][3]
        if(dr == 0 and dc == 0):
            continue

        pos_r = sink_r+dr
        pos_c = sink_c+dc
        while(valid(pos_r, pos_c) and (usr_brd[pos_r][pos_c]==1 or usr_brd[pos_r][pos_c]==3)): # it's a hit, NOT another sink - and note that it COULD be a FULLY placed ship already as well! 1 or 3!
            pos_r+=dr
            pos_c+=dc
        # undo going one extra
        pos_r-=dr
        pos_c-=dc

        dr_list = [-1, 1, 0, 0]
        dc_list = [0, 0, -1, 1]
        enclosed = True
        for dir in range(len(dr_list)):
            if((dr_list[dir], dc_list[dir])==(-dr, -dc)): # ignore the ship body from which you are coming, obviously (opposite direction of dr / dc steps)
                # print((dr_list[dir], dc_list[dir]), (-dr, -dc))
                continue
            adjpos_r = pos_r+dr_list[dir]
            adjpos_c = pos_c+dc_list[dir]
            if(valid(adjpos_r, adjpos_c) and usr_brd[adjpos_r][adjpos_c]!=0): # could actually place it there / doesn't meet condition
                enclosed=False

        if(enclosed):
            new_fs_ctr+=1
            fixed_ship[fs_ctr][0]=sink_r
            fixed_ship[fs_ctr][1]=sink_c
            fixed_ship[fs_ctr][2]=pos_r
            fixed_ship[fs_ctr][3]=pos_c
            fixed_ship[fs_ctr][4]=max(abs(sink_r-pos_r)+1, abs(sink_c-pos_c)+1)

    if(new_fs_ctr > fs_ctr):
        print("NEW SHIP(S) PLACED! full data:")
        print(fixed_ship)
        print(sink_data)
        fs_ctr = new_fs_ctr
        post_fix_resample(usr_brd)
        return True


    return False

# called once you have placed a new ship (as per evaluate_fixed_ship)
def post_fix_resample(usr_brd):
    global fs_ctr
    global fixed_ship

    for i in range(fs_ctr):
        end1_r, end1_c, end2_r, end2_c, len = fixed_ship[i][0], fixed_ship[i][1], fixed_ship[i][2], fixed_ship[i][3], fixed_ship[i][4]
        min_r = min(end1_r, end2_r)
        max_r = max(end1_r, end2_r)
        min_c = min(end1_c, end2_c)
        max_c = max(end1_c, end2_c)

        # place the ship on user board
        if(min_r == max_r):
            for col in range(min_c, max_c+1):
                usr_brd[min_r][col]=3 # 3 now represents a FULLY PLACED SHIP
        else: # ship_c_min == ship_c_max
            for row in range(min_r, max_r+1):
                usr_brd[row][min_c]=3

    #now, trigger the (re)sampling routine *WITH THE GIVEN USER_BOARD STATE*
    gen_n_samples(N_SAMPLES, usr_brd)
    reset_sample_dist_list(usr_brd)
    if(VERBOSE):
        print(N_SAMPLES, "new samples generated: here is example of closest-distance board")
        print(SAMPLE_ARRAY[SORTED_INDS[0][1]])

# called by post_fix_resample to re-evaluate distances of all new board based on past queries
def reset_sample_dist_list(usr_brd):
    global SORTED_INDS
    global IND_VALUES
    global N_SAMPLES
    global query_log
    global GUESSED_SET

    # RESET THESE DATA STRUCTURES
    SORTED_INDS = SortedList()
    IND_VALUES = {}

    num_queries = len(GUESSED_SET)

    for i in range(N_SAMPLES):
        dist = 0
        for q in range(num_queries):
            q_r, q_c, q_val = query_log[q][0], query_log[q][1], query_log[q][2]
            sample_val = SAMPLE_ARRAY[i][q_r][q_c]

            if((q_val==1 and sample_val==3) or (q_val==2 and sample_val==3) or (q_val==1 and sample_val==2)):
                continue
            # if(q_val == 0 or (q_val==1 and sample_val==3) or (q_val==2 and sample_val==3) or (q_val==1 and sample_val==2)):
            #     continue


            # CASE_WORK:
            #   (DEPR: NO LONGER TRUE: if q_val = 0, then ALL sample_boards currently have that miss.)
            #   if q_val = 1 or 2 and NOW it's a 3 on usr_brd, then sample_board ALSO has that correct.
            #   so only need to update non-3 1s that are NOT 1s or 2s ; non-3 2s that are NOT 2s

            # need to incorporate q=0 now: misses matter
            elif(q_val != sample_val):
                # print(q_val, sample_val)
                dist+=1

        SORTED_INDS.add((dist, i))
        IND_VALUES[i]=dist

# updates user's board and propagate the new result into data structures
# then calls update_all_sample_dist...
def update_user_board(usr_brd, guess_set, r, c, response_val):
    # 0 = miss, 1 = hit
    usr_brd[r][c]=response_val
    query_log[len(guess_set)][0]=r
    query_log[len(guess_set)][1]=c
    query_log[len(guess_set)][2]=response_val
    guess_set.add(grid_ind(r,c))

    update_all_sample_dist(r, c, response_val)

    new_fixed = evaluate_fixed_ship(usr_brd)

# based on the update in update_user_board, updates distances of all sample board from the new board state
def update_all_sample_dist(r, c, response_val):
    global N_SAMPLES
    global SAMPLE_ARRAY

    for i in range(N_SAMPLES):
        sample_board_val = SAMPLE_ARRAY[i][r][c]

        if(response_val == 1 and sample_board_val==2):
            continue
        elif(response_val != sample_board_val):
            update_dist(i, 1)

# when get a hit at (r,c), propagate updates through this function (whether we're starting active hit, going down a line, etc)
def hit_updates(r, c):
    global ACTIVE_HIT
    global LINE_FOUND
    global line_min_loc
    global line_max_loc

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

# when get a sink, propagate updates through this function (check for game end, call update_sink_data to store where we have sinks)
def sink_updates(r, c):
    global SINK_CTR
    global ACTIVE_HIT
    global LINE_FOUND
    global END_FLAG

    global hit_loc
    global line_max_loc
    global line_min_loc

    SINK_CTR += 1
    if(SINK_CTR == 5):
        END_FLAG = True
        return

    update_sink_data(r, c, SINK_CTR)

    ACTIVE_HIT = False
    LINE_FOUND = False

    # THIS IS NEW: to be tested
    hit_loc=[None, None]
    line_max_loc=[None, None]
    line_min_loc=[None, None]

# adds data about this sink (and the direction from which it came) to data structure
def update_sink_data(sink_r, sink_c, sink_num):
    global sink_data
    global hit_loc
    global line_max_loc
    global line_min_loc

    # NOW SINK_NUM is NEW: so preprocess and decrement
    sink_num-=1

    sink_data[sink_num][0]=sink_r
    sink_data[sink_num][1]=sink_c

    dr = None
    dc = None

    if(LINE_FOUND):
        if(line_max_loc[0]==sink_r):
            if(sink_c >= line_max_loc[1]):
                dr=0
                dc=-1
            else:
                dr=0
                dc=1
        elif(line_max_loc[1]==sink_c):
            if(sink_r >= line_max_loc[0]):
                dr=-1
                dc=0
            else:
                dr=1
                dc=0
        else:
            print("Error in checking sink origin: sink loc is", to_one_index(sink_r, sink_c), "while line locs are", to_one_index(line_min_loc), to_one_index(line_max_loc))

    elif(ACTIVE_HIT and not (hit_loc[0]==sink_r and hit_loc[1]==sink_c)):
        if(hit_loc[0]==sink_r and hit_loc[1] < sink_c):
            dr=0
            dc=-1
        elif(hit_loc[0]==sink_r and hit_loc[1] > sink_c):
            dr=0
            dc=1
        elif(hit_loc[0] < sink_r and hit_loc[1] == sink_c):
            dr=-1
            dc=0
        else:
            dr=1
            dc=0
    else:
        dr=0
        dc=0
        print("Hmm...no clear line found for this sink?")

    sink_data[sink_num][2]=dr
    sink_data[sink_num][3]=dc

    return

# helper method for update_all_sample_dist
def update_dist(ind, delta):
    # hopefully this is speedy enough?
    global SORTED_INDS
    global IND_VALUES

    val = IND_VALUES[ind]
    SORTED_INDS.remove((val, ind))
    IND_VALUES[ind]+=delta
    SORTED_INDS.add((val+delta, ind))

# determine guess based on sampling procedure, by evaluating some number of most similar boards (defined by top_sample_thresh)
def find_guess(top_sample_thresh, guess_set):
    global SORTED_INDS
    global IND_VALUES

    master_counts = np.zeros((BOARD_SZ, BOARD_SZ), dtype='int')

    max_count = 0
    max_count_r = -1
    max_count_c = -1

    unguessed_set = set()

    thresh=top_sample_thresh

    num_zero_dist = 0
    while(num_zero_dist<N_SAMPLES and SORTED_INDS[num_zero_dist][0]==0):
        num_zero_dist+=1

    if(VERBOSE):
        print('num 0 dist boards:', num_zero_dist)

    if(num_zero_dist>0):
        thresh = min(MAX_THRESH, num_zero_dist)

    # if(num_zero_dist==1):
    #     print_board(SAMPLE_ARRAY[SORTED_INDS[0][1]])


    for r in range(BOARD_SZ):
        for c in range(BOARD_SZ):
            grid_rc = grid_ind(r, c)
            if(grid_rc in guess_set):
                continue

            unguessed_set.add(grid_rc)

            for i in range(thresh):
                indd = SORTED_INDS[i][1] # THE KEY LINE! - we want *what index the ith in SORTED LIST corresponds to*
                if(SAMPLE_ARRAY[indd][r][c]==1): #hit/has ship
                    master_counts[r][c]+=1

            if(master_counts[r][c] > max_count): # keep a running max
                max_count = master_counts[r][c]
                max_count_r = r
                max_count_c = c


    if(max_count > 0):
        return max_count_r, max_count_c

    else:
        print("Going with random guess...")
        rand_guess = random.sample(unguessed_set,1)[0]
        return int(rand_guess/10), rand_guess%10

# determines whether to (1) do sampling-based guess, (2) assess adjacent squares after first hit, or (3) track down a line
def find_move(top_sample_thresh):
    global ACTIVE_HIT
    global LINE_FOUND
    global GUESSED_SET
    global USER_BOARD

    if(not ACTIVE_HIT and not LINE_FOUND):
        if(VERBOSE):
            print("CASE 1: SAMPLE-BASED GUESSING")
        return find_guess(top_sample_thresh, GUESSED_SET)

    elif(ACTIVE_HIT and not LINE_FOUND):
        if(VERBOSE):
            print("CASE 2: ACTIVE_HIT + ASSESS_ADJ:", to_one_index(hit_loc))
        return assess_adj(USER_BOARD, hit_loc)

    else: #ACTIVE_HIT and LINE_FOUND both True
        if(VERBOSE):
            print("CASE 3: LINE GUESSING: min:", to_one_index(line_min_loc), "; max:", to_one_index(line_max_loc))
        return guess_line(USER_BOARD, hit_loc, line_min_loc, line_max_loc)

# helper method to print user-end data in 1-indexed coordinates
def to_one_index(loc):
    return (loc[0]+1, loc[1]+1)

## --------- USER END FUNCTIONS ----------- #

# print the board out with 'H'=hit / 'X'=miss / '_'=unguessed / 'S'=sink / '+'=ship placed for sure, length determined
#  and using conventions of game [(1,1) is left-bottom, (10, 10) is right-top]
def print_board(brd):
    # NOTE: THIS PRINTS FLIPPED BOARD – because of (1, 1) for LEFT-BOTTOM ; (10, 10) for RIGHT-TOP
    # we want the order to be in the user's request order, and then compute the internal r_comp and c_comp from that
    for r_u in range(BOARD_SZ):
        str = ''
        for c_u in range(BOARD_SZ):
            r = c_u
            c = BOARD_SZ-r_u-1
            if(brd[r][c]==-1):
                str+='_'
            elif(brd[r][c]==0):
                str+='X'
            elif(brd[r][c]==1):
                str+='H'
            elif(brd[r][c]==2):
                str+='S'
            else: # for 3, for a fully placed ship
                str+='+'
        print(str)

# run the game!
def play_game(top_sample_thresh):
    global GUESSED_SET
    global USER_BOARD
    global END_FLAG

    global N_SAMPLES
    global MAX_THRESH
    global TOP_THRESH
    global VERBOSE

    print("welcome to battleship.")
    print("COMMANDS: 0 for 'Miss', 1, for 'Hit', 2 for 'Hit and Sunk'")
    print("KEY PARAMETERS: (1) number of initial samples, (2) max. number of samples to consider per move, (3) min. number of (non-zero) distance samples to consider per move. Type -1 for defaults.")
    user_nsamples = int(input("1. number of samples? (default: " + str(N_SAMPLES)+")"))
    user_maxthresh = int(input("2. max. num. samples per move? (default: " + str(MAX_THRESH)+")"))
    user_topthresh = int(input("3. min. num. of non-zero-dist samples per move? (default: " + str(TOP_THRESH)+")"))
    user_verbose = int(input("4. Verbose? 1 for yes, anything other integer for no."))

    if(user_nsamples!=-1):
        N_SAMPLES=user_nsamples
    if(user_maxthresh!=-1):
        MAX_THRESH=user_maxthresh
    if(user_topthresh!=-1):
        TOP_THRESH=user_topthresh
    if(user_verbose==1):
        VERBOSE=True



    gen_n_samples(N_SAMPLES, USER_BOARD)
    # print(SAMPLE_ARRAY)
    # for i in range(N_SAMPLES):
    #     print_board(SAMPLE_ARRAY[i])
    #     print()

    command = "start"
    FIRST_FLAG = True
    # prev_r = -1
    # prev_c = -1

    sink_ctr = 0
    while(command!='STOP'):
        # 0 for miss, 1 for hit, 2 for hit and sunk
        if(not FIRST_FLAG):
            comm_int = int(command)
            if(comm_int==1):
                hit_updates(r, c)
                update_user_board(USER_BOARD, GUESSED_SET, r, c, 1)
            elif(comm_int==2):
                sink_updates(r,c)
                update_user_board(USER_BOARD, GUESSED_SET, r, c, 2) # no differentiation between "hit" and "hit and sunk" here
            elif(comm_int==0):
                update_user_board(USER_BOARD, GUESSED_SET, r, c, 0)
            else:
                print("Please enter a valid guess.")


        FIRST_FLAG = False
        print_board(USER_BOARD)
        if(END_FLAG):
            print("Game over in", len(GUESSED_SET), "moves!")
            return


        # print(SORTED_INDS)
        if(VERBOSE):
            print("closest board distance", SORTED_INDS[0][0])

        r, c = find_move(top_sample_thresh)
        print("Query:", to_one_index((r, c))) # 1-index!!
        # prev_r = r
        # prev_c = c

        print("Guesses so far (including this query):", len(GUESSED_SET)+1) # because this query has not actually been placed in guessed_set yet
        # print(len(SORTED_INDS))
        command=input("command: ")
        print("\n")


play_game(TOP_THRESH)
