if __name__ == '__main__':
    with open("input.txt") as f:
        min_input = None
        min_cost = 1e16

        lines = f.readlines()

        for i in range(0, len(lines), 2):
            input_line = lines[i]
            cost_line = lines[i+1]
            cost = float(cost_line[6:])

            if cost < min_cost:
                min_input = input_line
                min_cost = cost

        print(min_input, min_cost)
