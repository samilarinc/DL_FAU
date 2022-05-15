from sklearn.utils import shuffle
from generator import ImageGenerator

batch_size = 17
rot = False
mir = False
shuf = False
out_size = 40,40,3

gen = ImageGenerator("data/exercise_data/", "data/Labels.json", batch_size, (32,32,3), rot, mir, shuf, out_size=out_size)

gen.show()