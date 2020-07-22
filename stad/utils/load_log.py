def load_log():

    with open("run.log", "r") as f:
        lines = f.readlines()

    di = {}
    for line in lines:
        line = line.replace("\n", "")
        _, k, v = line.split(" - ")

        if k == "epoch":
            epoch = v
            di[epoch] = {"h": [], "w": [], "loss": []}

        elif k == "loss":
            di[epoch][k] = float(v)

        elif k in ["h", "w"]:
            di[epoch][k].append(float(v))

    return di
