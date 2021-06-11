import json
import json

file_in = json.load(open("./mycode/Traindata/aicm2/VID000_0/point_labels/sim_000000.json"))
points = file_in["points"]
for point in points:
    x = point["x"]
    y = point["y"]
    print("x:",x,"y:",y)