from sklearn.utils import shuffle
from generator import ImageGenerator

batch_size = 17
rot = False
mir = False
shuf = False
out_size = None

gen = ImageGenerator("data/exercise_data/", "data/Labels.json", 10, (32,32,3), rot, mir, shuf, out_size=out_size)