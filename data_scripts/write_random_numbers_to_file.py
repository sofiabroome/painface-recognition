import sys
sys.path.append('..')
import random
import helpers

NB_PARTICIPANTS = 50

entries = []
randnlist = random.sample(range(300, 500), NB_PARTICIPANTS)

randnlist = [str(x) for x in randnlist]

helpers.write_file('sifferkoder.txt', randnlist)
