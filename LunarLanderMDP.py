# Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl
import numpy as np

class Grid: # Environment
  def __init__(self, width, height, start):
    self.width = width
    self.height = height
    self.i = start[0]
    self.j = start[1]

  def set(self, rewards, actions):
    # rewards should be a dict of: (i, j): r (row, col): reward
    # actions should be a dict of: (i, j): A (row, col): list of possible actions
    self.rewards = rewards
    self.actions = actions

  def set_state(self, s):
    self.i = s[0]
    self.j = s[1]

  def current_state(self):
    return (self.i, self.j)

  def is_terminal(self, s):
    return s not in self.actions

  def move(self, action):
    # check if legal move first
    #actions are: do nothing 'N', go down 'D', go left 'L', go right 'R'
    if action in self.actions[(self.i, self.j)]:
      if action == 'N':
        self.i =self.i
        self.j=self.j
      elif action == 'D':
        self.i += 1
      elif action == 'R':
        self.j += 1
      elif action == 'L':
        self.j -= 1
    # return a reward (if any)
    return self.rewards.get((self.i, self.j), 0)

  def undo_move(self, action):
    # these are the opposite of what N/D/L/R should normally do
    if action == 'N':
      self.i = self.i
      self.j=self.j
    elif action == 'D':
      self.i -= 1
    elif action == 'R':
      self.j -= 1
    elif action == 'L':
      self.j += 1
    # raise an exception if we arrive somewhere we shouldn't be
    # should never happen
    assert(self.current_state() in self.all_states())

  def game_over(self):
    # returns true if game is over, else false
    # true if we are in a state where no actions are possible
    return (self.i, self.j) not in self.actions

  def all_states(self):
    # possibly buggy but simple way to get all states
    # either a position that has possible next actions
    # or a position that yields a reward
    return set(self.actions.keys()) | set(self.rewards.keys())


def standard_grid():
  # define a grid that describes the reward for arriving at each state
  # and possible actions at each state
  # the grid looks like this
  # s means start position. start position is at (0,2)
  # number means reward at that state
  #   .     .     s     .     .
  #   .     .     .     .     .
  # -100  -100  +220  -100  -100
  g = Grid(3, 5, (0, 2))
  rewards = {(2, 0): -100, (2, 1): -100,(2,2):220,(2,3):-100,(2,4):-100}
  #actions to take in every part of the grid except row 2 onwards
  actions = {
    (0, 0): ('D', 'R', 'N'),
    (0, 1): ('L', 'R', 'D', 'N'),
    (0, 2): ('L', 'R', 'D', 'N'),
    (0, 3): ('L', 'R', 'D', 'N'),
    (0, 4): ('D', 'L', 'N'),
    (1, 0): ('D', 'R', 'N'),
    (1, 1): ('L', 'R', 'D', 'N'),
    (1, 2): ('L', 'R', 'D', 'N'),
    (1, 3): ('L', 'R', 'D', 'N'),
    (1, 4): ('D', 'L', 'N'),
  }
  g.set(rewards, actions)
  return g


def negative_grid(step_cost=-0.3):
  # penalize for using the engine - every move will be penalized
  # so we will penalize every move
  g = standard_grid()
  g.rewards.update({
    (0, 0): step_cost,
    (0, 1): step_cost,
    (0, 2): step_cost,
    (0, 3): step_cost,
    (0, 4): step_cost,
    (1, 0): step_cost,
    (1, 1): step_cost,
    (1, 2): step_cost,
    (1, 3): step_cost,
    (1, 4): step_cost,
  })
  return g


def print_values(V, g):
  for i in range(g.width):
    print("---------------------------")
    for j in range(g.height):
      v = V.get((i,j), 0)
      if v >= 0:
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="") # -ve sign takes up an extra space
    print("")


def print_policy(P, g):
  for i in range(g.width):
    print("---------------------------")
    for j in range(g.height):
      a = P.get((i,j), ' ')
      print("  %s  |" % a, end="")
    print("")