from __future__ import print_function

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

import numpy as np

from PIL import Image, ImageDraw, ImageFont

import itertools
import logging
from six import StringIO


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

class IllegalMove(Exception):
    pass

class test_illegal(Exception):
    pass

def stack(flat, layers=16):
    """Convert an [4, 4] representation into [layers, 4, 4] with one layers for each value."""
    # representation is what each layer represents
    representation = 2 ** (np.arange(layers, dtype=int) + 1)

    # layered is the flat board repeated layers times
    layered = np.repeat(flat[:,:,np.newaxis], layers, axis=-1)

    # Now set the values in the board to 1 or zero depending whether they match representation.
    # Representation is broadcast across a number of axes
    layered = np.where(layered == representation, 1, 0)
    layered = np.transpose(layered, (2,0,1))
    return layered

class My2048Env(gym.Env):
    metadata = {
        "render_modes": ['ansi', 'human', 'rgb_array'],
        "render_fps": 2,
    }

    def __init__(self):
        # Definitions for game. Board must be square.
        self.size = 4
        self.w = self.size
        self.h = self.size
        self.squares = self.size * self.size

        # Maintain own idea of game score, separate from rewards
        self.score = 0

        # Foul counts for illegal moves
        self.foul_count = 0

        # Members for gym implementation
        self.action_space = spaces.Discrete(4)
        # Suppose that the maximum tile is as if you have powers of 2 across the board.
        layers = self.squares
        self.observation_space = spaces.Box(0, 1, (layers, self.w, self.h), dtype=int)
        
        # TODO: Set negative reward (penalty) for illegal moves (optional)
        self.set_illegal_move_reward(-8.0)
        
        self.set_max_tile(None)

        # Size of square for rendering
        self.grid_size = 70

        # Initialise seed
        self.seed()

        # Reset ready for a game
        self.reset()

        #remember the step_count
        self.step_count = 0

        #remember previous action
        self.pre_act = None

        #remember current highest in an episode
        self.current_highest = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_illegal_move_reward(self, reward):
        """Define the reward/penalty for performing an illegal move. Also need
            to update the reward range for this."""
        # Guess that the maximum reward is also 2**squares though you'll probably never get that.
        # (assume that illegal move reward is the lowest value that can be returned
        self.illegal_move_reward = reward
        self.reward_range = (self.illegal_move_reward, float(2**self.squares))

    def set_max_tile(self, max_tile):
        """Define the maximum tile that will end the game (e.g. 2048). None means no limit.
           This does not affect the state returned."""
        assert max_tile is None or isinstance(max_tile, int)
        self.max_tile = max_tile

    def is_n_and_half_n_ajacent(self,n): #n=128,256,512,1024
        # if self.highest()<n:
        #     return False
        # else:
        try:
            highest_is,highest_js = np.where(self.Matrix==n) #find n's index
            highest_i = highest_is[0]
            highest_j = highest_js[0]

            sec_highest_is,sec_highest_js = np.where(self.Matrix==(n/2)) #find n's index
            sec_highest_i = sec_highest_is[0]
            sec_highest_j = sec_highest_js[0]
        except:
            return False

        if len(highest_is)>=2 and len(sec_highest_is)>=2: #no bonus for second n/2 block
            return False

        if (highest_i==sec_highest_i and abs(highest_j - sec_highest_j)==1) or (abs(highest_i-sec_highest_i)==1 and highest_j == sec_highest_j):
            return True
        else:
            return False
        
    def first_block_n(self,n):#detect block with value n is the first block in self.Matrix
        try:
            highest_is,highest_js = np.where(self.Matrix==n) #find n's index
        except:
            return False
        
        if len(highest_is)==1:
            return True
        elif len(highest_is)>1:
            return False
            

    # Implement gym interface
    def step(self, action):
        """Perform one step of the game. This involves moving and adding a new tile."""
        logging.debug("Action {}".format(action))
        score = 0
        done = None
        info = {
            'illegal_move': False,
            'highest': 0,
            'score': 0,
        }
        self.step_count += 1
        # print(self.step_count, self.Matrix)
        try:
            # assert info['illegal_move'] == False
            pre_state = self.Matrix.copy()
            score = float(self.move(action))
            self.score += score
            assert score <= 2**(self.w*self.h)
            self.add_tile()
            done = self.isend()
            reward = float(score)

            # TODO: Add reward according to weighted states (optional)
            # weight = np.array([
            #         [4.0  , 2.0  , 1.75  , 0.1  ],
            #         [3.75  , 2.25  , 1.5  , 0.25  ],
            #         [3.5  , 2.5  , 1.25  , 0.5  ],
            #         [3.0  , 2.75  , 1.0  , 0.75  ]])
            # weight = np.array([
            #         [8.0  , 2.0  , 2.0  , 0.25  ],
            #         [8.0  , 2.0  , 1.0  , 0.5  ],
            #         [8.0  , 4.0  , 1.0  , 0.5  ],
            #         [4.0  , 4.0  , 1.0  , 0.5  ]])
            weight = np.array([
                    [16.0  , 1.0  , 1.0  , -8.0  ],
                    [8.0  , 1.0  , -1.0  , -8.0  ],
                    [8.0  , 4.0  , -1.0  , -4.0  ],
                    [4.0  , 4.0  , -2.0  , -4.0  ]])
            # reward += 0
            state_bonus = self.Matrix * weight #put the block as a snake from big to small help find the best solution
            # print(f"self.highest:{self.highest()}, state_bonus:",np.sum(state_bonus) * 0.004 )
            # if self.highest()>=512:
            #     if self.highest()>=1024:
            #         print("1024 merged!!!", np.sum(state_bonus) * 0.0009) #0.0025, when create 256, bonus=11
            #     else:
            #         print(np.sum(state_bonus) * 0.0009)
            if self.highest()>=1024:
                print("1024 merged!!!", np.sum(state_bonus) * 0.0009) #0.0025, when create 256, bonus=11
            reward += np.sum(state_bonus) * 0.0009 

            #reward for putting the biggest block at the corner
            # if self.Matrix[0,0]==self.highest() and self.highest()>=256:
            # #    reward += np.sum(state_bonus) * 0.05 
            #     reward += np.log2(self.highest())*0.05

            #we expect the block order above, penalty for using action : right, down, if >256 block is merged, keep it at the corner
            if action==3 and self.highest()>=512: #severly avoid use right
                reward-=3.0
                if self.highest()>=512:
                    reward-=2.0
            # elif action==1 and self.highest()>=256: #slightly avoid use down
            #     reward-=0.5
            #     if self.highest()>=512:
            #         reward-=0.5


            #Bonus for create consecutive blocks with the same number
            #we only consider the biggest block
            # highest_i,highest_j = np.where(self.Matrix==self.highest())
            # highest_i = highest_i[0]
            # highest_j = highest_j[0]
            # if self.Matrix[highest_i,highest_j]==self.Matrix[min(highest_i+1,3),highest_j]:
            #     reward += np.log2(self.highest()) * 0.2
            # elif self.Matrix[highest_i,highest_j]==self.Matrix[highest_i,min(highest_j+1,3)]:
            #     reward += np.log2(self.highest()) * 0.2
            # elif self.Matrix[highest_i,highest_j]==self.Matrix[max(highest_i-1,0),highest_j]:
            #     reward += np.log2(self.highest()) * 0.2
            # elif self.Matrix[highest_i,highest_j]==self.Matrix[highest_i,max(highest_j-1,0)]:
            #     reward += np.log2(self.highest()) * 0.2
            

            #Bonus for create consecutive blocks with 1024,512,256,128,64,32,16,8,4,2
            #we first create a list save the result of the matrix
            # block_list = [] # = [(0,0),(1,0),(2,0),(3,0),(3,1),...] of self.Matrix
            # block_list.extend(self.Matrix[:, 0])
            # block_list.extend(self.Matrix[::-1, 1])
            # block_list.extend(self.Matrix[:, 2])
            # block_list.extend(self.Matrix[::-1, 3])


            #we only detect the sequence starting from the biggest block
            # seq_begin = block_list.index(self.highest())
            # seq_length=0
            # while True:
            #     if seq_begin+seq_length+1 > len(block_list)-1:
            #         break
            #     if block_list[seq_begin+seq_length+1]==block_list[seq_begin+seq_length]/2:
            #         reward += np.log2(block_list[seq_begin+seq_length]) * 0.5
            #         seq_length += 1
            #     else:
            #         break
            


            # #Bonus for create 256
            # if self.highest()>=256:
            #     reward+=3.0

            # #Bonus for create 512
            # if self.highest()>=512:
            #     reward+=8.0

            # #Bonus for create 1024
            # if self.highest()>=1024:
            #     reward+=10.0

            # if self.highest()>=2048:
            #     reward+=1000.0
            #     self.step_count = 0
            #     done=True

            self.pre_act = action
            
        except IllegalMove:
            logging.debug("Illegal move")
            info['illegal_move'] = True
            reward = self.illegal_move_reward

            # TODO: Modify this part for the agent to have a chance to explore other actions (optional)
            if np.all(self.Matrix!=0): #then you successfully fill up the board, which should be rewarded
                if self.highest()>=512:
                    print(f"full done!, after create {self.highest()}")
                reward += 1.0
                self.step_count = 0
                done = True
            elif self.foul_count > 30:# and self.foul_count>self.step_count: 16
                if self.highest()>=512:
                    print(f"Too many illegal move!, after create {self.highest()}")
                reward -= 5.0
                self.step_count = 0
                done = True
            else:
                self.foul_count+=1

        truncate = False
        info['highest'] = self.highest()
        info['score']   = self.score

        if self.highest()>=512:
            self.set_illegal_move_reward(-8.0)

        # Return observation (board state), reward, done, truncate and info dict
        return stack(self.Matrix), reward, done, truncate, info

    def reset(self, seed=None, options=None):
        self.seed(seed=seed)
        self.Matrix = np.zeros((self.h, self.w), int)
        self.score = 0
        self.foul_count = 0

        # self.step_count = 0

        logging.debug("Adding tiles")
        self.add_tile()
        self.add_tile()

        return stack(self.Matrix), {}

    def render(self, mode='ansi'):
        outfile = StringIO() if mode == 'ansi' else None
        s = 'Score: {}\n'.format(self.score)
        s += 'Highest: {}\n'.format(self.highest())
        npa = np.array(self.Matrix)
        grid = npa.reshape((self.size, self.size))
        s += "{}\n".format(grid)
        outfile.write(s)
        return outfile

    # Implement 2048 game
    def add_tile(self):
        """Add a tile, probably a 2 but maybe a 4"""
        possible_tiles = np.array([2, 4])
        tile_probabilities = np.array([0.9, 0.1])
        val = self.np_random.choice(possible_tiles, 1, p=tile_probabilities)[0]
        empties = self.empties()
        assert empties.shape[0]
        empty_idx = self.np_random.choice(empties.shape[0])
        empty = empties[empty_idx]
        logging.debug("Adding %s at %s", val, (empty[0], empty[1]))
        self.set(empty[0], empty[1], val)

    def get(self, x, y):
        """Return the value of one square."""
        return self.Matrix[x, y]

    def set(self, x, y, val):
        """Set the value of one square."""
        self.Matrix[x, y] = val

    def empties(self):
        """Return a 2d numpy array with the location of empty squares."""
        return np.argwhere(self.Matrix == 0)

    def highest(self):
        """Report the highest tile on the board."""
        return np.max(self.Matrix)

    def move(self, direction, trial=False):
        """Perform one move of the game. Shift things to one side then,
        combine. directions 0, 1, 2, 3 are up, right, down, left.
        Returns the score that [would have] got."""
        if not trial:
            if direction == 0:
                logging.debug("Up")
            elif direction == 1:
                logging.debug("Right")
            elif direction == 2:
                logging.debug("Down")
            elif direction == 3:
                logging.debug("Left")

        changed = False
        move_score = 0
        dir_div_two = int(direction / 2)
        dir_mod_two = int(direction % 2)
        shift_direction = dir_mod_two ^ dir_div_two # 0 for towards up left, 1 for towards bottom right

        # Construct a range for extracting row/column into a list
        rx = list(range(self.w))
        ry = list(range(self.h))

        if dir_mod_two == 0:
            # Up or down, split into columns
            for y in range(self.h):
                old = [self.get(x, y) for x in rx]
                (new, ms) = self.shift(old, shift_direction)
                # move_score += ms
                if not ms==0:
                    #bonus for create big block
                    if self.highest()==256:
                        if ms>=128:
                            if self.is_n_and_half_n_ajacent(256):
                                # print("merge 128 next to 256!")
                                # print(self.Matrix)
                                # print("--------------------------------------")
                                move_score += (np.log2(ms)+3.0)*1.1
                            else:
                                move_score += (np.log2(ms)+3.0)
                        elif ms>=64:
                            if self.is_n_and_half_n_ajacent(256) and self.is_n_and_half_n_ajacent(128):
                                move_score += (np.log2(ms)+3.0)*1.2
                            else:
                                move_score += (np.log2(ms)+3.0)
                        else:
                            move_score += (np.log2(ms)+3.0)
                    elif self.highest()==512:
                        if ms>=256:
                            if self.is_n_and_half_n_ajacent(512):
                                print("merge 256 next to 512!")
                                print(self.Matrix)
                                print("--------------------------------------")
                                move_score += (np.log2(ms)+6.0)*1.2
                            else:
                                move_score += (np.log2(ms)+6.0)
                        elif ms>=128:
                            if self.is_n_and_half_n_ajacent(512) and self.is_n_and_half_n_ajacent(256) :
                                print("merge 128 next to 256, and 256 next to 512!")
                                print(self.Matrix)
                                print("--------------------------------------")
                                move_score += (np.log2(ms)+6.0)*1.4
                            else:
                                move_score += (np.log2(ms)+6.0)
                        else:
                            move_score += (np.log2(ms)+6.0)
                    elif self.highest()==1024:
                        if ms>=512:
                            if self.is_n_and_half_n_ajacent(1024):
                                print("merge 512 next to 1024!")
                                print(self.Matrix)
                                print("--------------------------------------")
                                move_score += (np.log2(ms)+10.0)*1.5
                            else:
                                move_score += (np.log2(ms)+10.0)
                        elif ms>=256:
                            if self.is_n_and_half_n_ajacent(1024) and self.is_n_and_half_n_ajacent(512):
                                    print("merge 256 next to 512, and 512 next to 1024!")
                                    print(self.Matrix)
                                    print("--------------------------------------")
                                    move_score += (np.log2(ms)+10.0)*1.8
                            else:
                                move_score += (np.log2(ms)+10.0)                            
                        else:
                            move_score += (np.log2(ms)+10.0)
                    else:
                        move_score += np.log2(ms)
                if old != new:
                    changed = True
                    if not trial:
                        for x in rx:
                            self.set(x, y, new[x])
        else:
            # Left or right, split into rows
            for x in range(self.w):
                old = [self.get(x, y) for y in ry]
                (new, ms) = self.shift(old, shift_direction)
                # move_score += ms
                if not ms==0:
                    #bonus for create big block
                    if self.highest()==256:
                        if ms>=128:
                            if self.is_n_and_half_n_ajacent(256):
                                # print("merge 128 next to 256!")
                                # print(self.Matrix)
                                # print("--------------------------------------")
                                move_score += (np.log2(ms)+3.0)*1.1
                            else:
                                move_score += (np.log2(ms)+3.0)
                        elif ms>=64:
                            if self.is_n_and_half_n_ajacent(256) and self.is_n_and_half_n_ajacent(128):
                                move_score += (np.log2(ms)+3.0)*1.2
                            else:
                                move_score += (np.log2(ms)+3.0)
                        else:
                            move_score += (np.log2(ms)+3.0)
                    elif self.highest()==512:
                        if ms>=256:
                            if self.is_n_and_half_n_ajacent(512):
                                print("merge 256 next to 512!")
                                print(self.Matrix)
                                print("--------------------------------------")
                                move_score += (np.log2(ms)+6.0)*1.2
                            else:
                                move_score += (np.log2(ms)+6.0)
                        elif ms>=128:
                            if self.is_n_and_half_n_ajacent(512) and self.is_n_and_half_n_ajacent(256) :
                                print("merge 128 next to 256, and 256 next to 512!")
                                print(self.Matrix)
                                print("--------------------------------------")
                                move_score += (np.log2(ms)+6.0)*1.4
                            else:
                                move_score += (np.log2(ms)+6.0)
                        else:
                            move_score += (np.log2(ms)+6.0)
                    elif self.highest()==1024:
                        if ms>=512:
                            if self.is_n_and_half_n_ajacent(1024):
                                print("merge 512 next to 1024!")
                                print(self.Matrix)
                                print("--------------------------------------")
                                move_score += (np.log2(ms)+10.0)*1.5
                            else:
                                move_score += (np.log2(ms)+10.0)
                        elif ms>=256:
                            if self.is_n_and_half_n_ajacent(1024) and self.is_n_and_half_n_ajacent(512):
                                    print("merge 256 next to 512, and 512 next to 1024!")
                                    print(self.Matrix)
                                    print("--------------------------------------")
                                    move_score += (np.log2(ms)+10.0)*1.8
                            else:
                                move_score += (np.log2(ms)+10.0)                            
                        else:
                            move_score += (np.log2(ms)+10.0)
                    else:
                        move_score += np.log2(ms)
                if old != new:
                    changed = True
                    if not trial:
                        for y in ry:
                            self.set(x, y, new[y])
        if changed != True:
            raise IllegalMove
        
        if self.current_highest<self.highest() and self.Matrix[0,0]==self.highest() and self.highest()>=256:
            if self.highest()==256:
                move_score += 3.0
            elif self.highest()==512:
                move_score+=6.0
            elif self.highest()>=1024:
                move_score+=10.0

            if self.highest()>=1024:
                print("merge big block at the corner!!")
                print(self.Matrix)
                print("--------------------------------------")

            self.currt_highest=self.highest()
        else:
            self.currt_highest=self.highest()

        # move_score = np.log2(move_score)
        return move_score

    def combine(self, shifted_row):
        """Combine same tiles when moving to one side. This function always
           shifts towards the left. Also count the score of combined tiles."""
        move_score = 0
        combined_row = [0] * self.size
        skip = False
        output_index = 0
        for p in pairwise(shifted_row):
            if skip:
                skip = False
                continue
            combined_row[output_index] = p[0]
            if p[0] == p[1]:
                combined_row[output_index] += p[1]
                move_score += p[0] + p[1]
                # Skip the next thing in the list.
                skip = True
            output_index += 1
        if shifted_row and not skip:
            combined_row[output_index] = shifted_row[-1]

        return (combined_row, move_score)

    def shift(self, row, direction):
        """Shift one row left (direction == 0) or right (direction == 1), combining if required."""
        length = len(row)
        assert length == self.size
        assert direction == 0 or direction == 1

        # Shift all non-zero digits up
        shifted_row = [i for i in row if i != 0]

        # Reverse list to handle shifting to the right
        if direction:
            shifted_row.reverse()

        (combined_row, move_score) = self.combine(shifted_row)

        # Reverse list to handle shifting to the right
        if direction:
            combined_row.reverse()

        assert len(combined_row) == self.size
        return (combined_row, move_score)

    def isend(self):
        """Has the game ended. Game ends if there is a tile equal to the limit
           or there are no legal moves. If there are empty spaces then there
           must be legal moves."""

        if self.max_tile is not None and self.highest() == self.max_tile:
            return True

        for direction in range(4):
            try:
                self.move(direction, trial=True)
                # Not the end if we can do any move
                return False
            except IllegalMove:
                pass
        return True

    def get_board(self):
        """Retrieve the whole board, useful for testing."""
        return self.Matrix

    def set_board(self, new_board):
        """Retrieve the whole board, useful for testing."""
        self.Matrix = new_board
