## ------- BATTLESHIP -------
## SIMULATION FILE for battleship: computer vs computer
##
## author: Arya Maheshwari
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
### ---- GLOBAL VARIABLES / HELPER FNs ---- ###

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


# USER_BOARD = np.zeros((BOARD_SZ, BOARD_SZ), dtype='U10')
USER_BOARD = np.zeros((BOARD_SZ, BOARD_SZ), dtype=int)

ship_ind_list = [0, 1, 2, 3, 4]
ship_ind_map = {'c':4, 'b':3, 'd':2, 's':1, 'p':0}

N = 1000 # for simulations

HORIZONTAL = np.zeros((len(SHIP_CHARS)), dtype='bool') # true if ship oriented horizontally, false otherwise (p, s, d, b, c)

COMPUTER_BOARD = None

# ------------------------------------------------------ #

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
# MUST BE AN INTEGER
TOP_THRESH = int(max(20, 0.0002 * N)) # 10 boards each time, after 0
MAX_THRESH = 50000
N_SAMPLES = 200000

SAMPLE_ARRAY = np.zeros((N_SAMPLES, BOARD_SZ, BOARD_SZ), dtype='int')

SORTED_INDS = SortedList()
IND_VALUES = {}
for i in range(N_SAMPLES):
    SORTED_INDS.add((0, i))
    IND_VALUES[i]=0


for r in range(BOARD_SZ):
    for c in range(BOARD_SZ):
        USER_BOARD[r][c]=-1 # to start off

GUESSED_SET = set() #0 to 99, 10*r + c to note if guessed or not


SINK_CTR = 0
END_FLAG = False

### ------------------------------------------------------------ ###
### -------------------- GAMEPLAY FUNCTIONS -------------------- ###

## DELETED SHIP_INDS FROM EVERYTHING: that never changes anymore... ##

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

# UPDATED VERSION
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

# UPDATED VERSION
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




# *** NEEDS TESTING ***
# default: tries to find spots where a 2-ship could be placed (in any direction) and randomly guesses
def default_guess(usr_brd):
    poss_guesses = {1:[], 2:[], 3:[], 4:[]}
    for row in range(BOARD_SZ):
        for col in range(BOARD_SZ):
            # search for whether (row, col) can be LEFT, RIGHT, UP, or DOWN – and add for each
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
# SET-BASED NOW
def gen_sample(ship_inds, array_ind):

    global BOARD_SZ
    global SHIP_LENS
    global SHIP_CHARS
    global SAMPLE_ARRAY

    '''
    usr_brd_copy: holds information about misses and hits on board - EDITABLE VERSION (i.e. can be used in simulating)
    ship_inds: indices (corresponding to SHIP_LENS / SHIP_CHARS) of ships still in play (0 = p --> 4 = c)
    '''

    # start_time = time.time()

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

                if(SAMPLE_ARRAY[array_ind][row][col]!=0): # THERE IS SOMETHING THERE
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
                SAMPLE_ARRAY[array_ind][new_i1][new_i2]=2
            else:
                SAMPLE_ARRAY[array_ind][new_i1][new_i2]=1 # 1 = mid-ship

            remove_ind = get_ind(new_i1, new_i2, i3)
            if(remove_ind in rem_states):
                rem_states.remove(remove_ind)

            remove_ind = get_ind(new_i1, new_i2, (i3+1)%2)
            if(remove_ind in rem_states):
                rem_states.remove(remove_ind)

    # print(SAMPLE_ARRAY[array_ind])
    return

def gen_n_samples(n):
    global USER_BOARD
    global ship_ind_list
    print("Generating initial sample of configurations...this could take a couple minutes.")

    start = time.time()


    for i in range(n):
        gen_sample(ship_ind_list, i)
        # if(i%(n/20)==0):
        #     print(i)

    end = time.time()
    print("gen sample total time:", end-start)

    return

def update_user_board(usr_brd, guess_set, r, c, response_val):
    # 0 = miss, 1 = hit
    usr_brd[r][c]=response_val
    guess_set.add(grid_ind(r,c))

    for i in range(N_SAMPLES):
        sample_board_val = SAMPLE_ARRAY[i][r][c]

        if(response_val == 1 and sample_board_val==2):
            continue
        elif(response_val != sample_board_val):
            update_dist(i, 1)
        #
        # if(response_val == 1 and sample_board_val==0):
        # # the one case where 2 options ok: if you get a hit (unsure whether endpoint or not), either hit or endpoint in board is fine
        #     update_dist(i, 1)
        # elif(response_val == 2 and sample_board_val==1):
        # # this is close, so only penalize by 0.5
        #     update_dist(i, 0.5) # WAIT: should be penalizing this fully ... --> NEW FRAMEWORK ABOVE
        # elif(response_val == 2 and sample_board_val==0):
        #     update_dist(i, 1)
        # elif(response_val== 0 and sample_board_val > 0):
        #     update_dist(i, 1)
        # update_dist(i, abs(response_val-sample_board_val)) # this is basically XOR!

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

def sink_updates(r, c):
    global SINK_CTR
    global ACTIVE_HIT
    global LINE_FOUND
    global END_FLAG

    SINK_CTR += 1
    ACTIVE_HIT = False
    LINE_FOUND = False

    if(SINK_CTR == 5):
        END_FLAG = True
        return

def update_dist(ind, delta):
    # hopefully this is speedy enough?
    global SORTED_INDS
    global IND_VALUES

    val = IND_VALUES[ind]
    SORTED_INDS.remove((val, ind))
    IND_VALUES[ind]+=delta
    SORTED_INDS.add((val+delta, ind))

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
                indd = SORTED_INDS[i][1] # THE KEY LINE! – we want *what index the ith in SORTED LIST corresponds to*
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

def find_move(top_sample_thresh):
    global ACTIVE_HIT
    global LINE_FOUND
    global GUESSED_SET
    global USER_BOARD

    if(not ACTIVE_HIT and not LINE_FOUND):
        print("CASE 1: SAMPLE-BASED GUESSING")
        return find_guess(top_sample_thresh, GUESSED_SET)

    elif(ACTIVE_HIT and not LINE_FOUND):
        print("CASE 2: ACTIVE_HIT + ASSESS_ADJ:", to_one_index(hit_loc))
        return assess_adj(USER_BOARD, hit_loc)

    else: #ACTIVE_HIT and LINE_FOUND both True
        print("CASE 3: LINE GUESSING: min:", to_one_index(line_min_loc), "; max:", to_one_index(line_max_loc))
        return guess_line(USER_BOARD, hit_loc, line_min_loc, line_max_loc)

def to_one_index(loc):
    return (loc[0]+1, loc[1]+1)

## --------- USER END FUNCTIONS ----------- #

def print_board(brd):
    for r in range(BOARD_SZ):
        str = ''
        for c in range(BOARD_SZ):
            if(brd[r][c]==-1):
                str+='_'
            elif(brd[r][c]==0):
                str+='X'
            elif(brd[r][c]==1):
                str+='H'
            else:
                str+='S'
        print(str)

def play_game(top_sample_thresh):
    global GUESSED_SET
    global USER_BOARD
    global END_FLAG

    global N_SAMPLES
    global MAX_THRESH
    global TOP_THRESH

    print("welcome to battleship.")
    print("COMMANDS: 0 for 'Miss', 1, for 'Hit', 2 for 'Hit and Sunk'")
    print("KEY PARAMETERS: (1) number of initial samples, (2) max. number of samples to consider per move, (3) min. number of (non-zero) distance samples to consider per move. Type -1 for defaults.")
    user_nsamples = int(input("1. number of samples? (default: " + str(N_SAMPLES)+")"))
    user_maxthresh = int(input("2. max. num. samples per move? (default: " + str(MAX_THRESH)+")"))
    user_topthresh = int(input("3. min. num. of non-zero-dist samples per move? (default: " + str(TOP_THRESH)+")"))

    if(user_nsamples!=-1):
        N_SAMPLES=user_nsamples
    if(user_maxthresh!=-1):
        MAX_THRESH=user_maxthresh
    if(user_topthresh!=-1):
        TOP_THRESH=user_topthresh



    gen_n_samples(N_SAMPLES)
    # for i in range(N_SAMPLES):
    #     print_board(SAMPLE_ARRAY[i])
    #     print()

    command = input("command: ")
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
        if(END_FLAG):
            print("Game over in", len(GUESSED_SET), "moves!")
            return

        print_board(USER_BOARD)
        # print(SORTED_INDS)
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



# OLD CODE
#
# # HELPERS
# def guess_vert_line(usr_brd, min_loc, max_loc, line_len):
#     global ACTIVE_HIT
#     global LINE_FOUND
#     global line_min_loc
#     global line_max_loc
#
#     '''
#     min_loc and max_loc will be (a, c) --> (b, c), where b > a
#     '''
#
#     c = min_loc[1]
#     up_loc = (min_loc[0]-1, c)
#     down_loc = (max_loc[0]+1, c)
#
#     max_configs = 0
#
#     up_cts = 0
#     down_cts = 0
#
#     up_max = True
#
#     for x in range(len(SHIP_LENS)): # lowest to greatest length
#         ship_len = SHIP_LENS[x]
#         ship_char = SHIP_CHARS[x]
#
#         for up_ind in range(up_loc[0]-ship_len+1, down_loc[0]): # the left index of ship could be anywhere between these two values (pre-additional checking)
#             VIABLE = True
#             for step in range(up_ind, up_ind+ship_len):
#                 # here, unlike assess_adj, we are chill with line hits that aren't the starting/original hit position
#                 if(not valid(hit_r, step) or usr_brd[hit_r][step]==0): # other placed ships will have lowercase letter, these are not valid; misses are X; only CURRENT HITS are 'H'
#                     VIABLE=False
#
#             if(VIABLE):
#                 if(up_ind <= up_loc[0]):
#                     up_cts+=1 # one config includes UP spot
#                     if(up_cts>max_configs):
#                         max_configs = up_cts
#                         up_max=True
#
#                 if(up_ind+ship_len-1 >= down_loc[0]):
#                     down_cts+=1 # one config includes DOWN spot
#                     if(down_cts>max_configs):
#                         max_configs = down_cts
#                         up_max=False
#
#     if(up_cts + down_cts == 0): # if they're both still at 0
#         # WE NEED TO FLIP DIRECTION! (i.e. we can't place any ships along this line anymore...)
#
#         # new, rare edge case: there might still be a horiz line active..
#         # minc, maxc = find_minmax_horiz(usr_brd, hit_loc[0], hit_loc[1])
#         # if(minc != maxc):
#         #     line_min_loc = [hit_loc[0], minc]
#         #     line_max_loc = [hit_loc[0], maxc]
#         #     return guess_horiz_line(usr_brd, line_min_loc, line_max_loc, maxc-minc+1, ship_inds)
#
#         LINE_FOUND = False
#         line_min_loc = hit_loc # NEW
#         line_max_loc = hit_loc # NEW
#
#         # this is its all-catcher...which should trigger assess_adj's default catcher as needed?
#         return assess_adj(usr_brd, hit_loc, ship_inds) # right? just rotate around original hit if still active; if that doesn't work
#
#     elif(up_cts > down_cts):
#         return up_loc[0], up_loc[1]
#     else:
#         return down_loc[0], down_loc[1]
# # direct analog to guess_vert_line
# def guess_horiz_line(usr_brd, min_loc, max_loc, line_len):
#     global ACTIVE_HIT
#     global LINE_FOUND
#     global line_min_loc
#     global line_max_loc
#
#
#     '''
#     min_loc and max_loc will be (r, a) --> (r, b), where b > a
#     '''
#
#     r = min_loc[0]
#     left_loc = (r, min_loc[1]-1)
#     right_loc = (r, max_loc[1]+1)
#
#     left_cts = 0
#     right_cts = 0
#
#     max_configs = 0
#
#     left_max = True
#
#     for x in range(len(SHIP_LENS)): # lowest to greatest length
#         ship_len = SHIP_LENS[x]
#         ship_char = SHIP_CHARS[x]
#
#         for left_ind in range(left_loc[1]-ship_len+1, right_loc[1]): # the left index of ship could be anywhere between these two values (pre-additional checking)
#             VIABLE = True
#             for step in range(left_ind, left_ind+ship_len):
#                 # here, unlike assess_adj, we are chill with line hits that aren't the starting/original hit position
#                 if(not valid(step, hit_c) or usr_brd[step][hit_c]==0):
#                     VIABLE=False
#
#             if(VIABLE):
#                 if(left_ind <= left_loc[1]):
#                     left_cts+=1 # one config includes UP spot
#                     if(left_cts>max_configs):
#                         max_configs = left_cts
#                         left_max=True
#
#                 if(left_ind+ship_len-1 >= right_loc[1]):
#                     right_cts+=1 # one config includes DOWN spot
#                     if(right_cts>max_configs):
#                         max_configs = right_cts
#                         left_max=False
#
#     if(left_cts + right_cts == 0): # if they're both still at 0
#         # WE NEED TO FLIP DIRECTION! (i.e. we can't place any ships along this line anymore...)
#
#         # # new, rare edge case: there might still be a vertical line active..
#         # minr, maxr = find_minmax_vert(usr_brd, hit_loc[0], hit_loc[1])
#         # if(minr != maxr):
#         #     line_min_loc = [minr, hit_loc[1]]
#         #     line_max_loc = [maxr, hit_loc[1]]
#         #     return guess_vert_line(usr_brd, line_min_loc, line_max_loc, maxr-minr+1, ship_inds)
#
#         LINE_FOUND = False
#         line_min_loc = hit_loc # NEW
#         line_max_loc = hit_loc # NEW
#
#         # this is its all-catcher...which should trigger assess_adj's default catcher as needed?
#         return assess_adj(usr_brd, hit_loc, ship_inds) # right? just rotate around original hit if still active; if that doesn't work
#
#     elif(left_cts > right_cts):
#         return left_loc[0], left_loc[1]
#     else:
#         return right_loc[0], right_loc[1]
