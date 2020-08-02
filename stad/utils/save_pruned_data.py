def save_pruned_data():

    with open("run.log", "r") as f:
        lines = f.readlines()

    pruned_data = []
    for line in lines:
        line = line.replace("\n", "")
        _, k, v = line.split(" - ")

        if k == "pruned data":
            pruned_data.append(v)

    with open("pruned_data.txt", "w") as f:
        f.write("\n".join(pruned_data))
