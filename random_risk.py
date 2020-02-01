import random
import json
from model import PathClass

dic = {}

for i in range(0, 256):
    dic[i] = random.random()
with open(PathClass.instance().getCorrelation(),"w",encoding="UTF-8") as f:
    f.write(json.dumps(dic))
    f.close()
