from sklearn.utils import shuffle
from generator import ImageGenerator

batch_size = 17
rot = False
mir = False
shuf = False

gen = ImageGenerator("exercise_data/", "Labels.json", 50, (32,32,3), rot, mir, shuf)
print(gen.current_epoch())
gen.next()
print(gen.current_epoch())
gen.next()
print(gen.current_epoch())
gen.next()
print(gen.current_epoch())
gen.next()
print(gen.current_epoch())