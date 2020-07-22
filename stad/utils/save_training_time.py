def save_training_time():

    with open("run.log", "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.replace("\n", "")
        _, k, v = line.split(" - ")

        if k == "training start":
            start = float(v)

        elif k == "training end":
            end = float(v)

    with open("training_time.txt", "w") as f:
        f.write(f"training_time - {start - end}")
