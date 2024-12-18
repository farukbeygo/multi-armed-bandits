
count = 0
old_estimate = 0.0

while True:
    count += 1
    val = float(input("Enter a number: "))
    current_estimate = old_estimate + (1/count) * (val - old_estimate)
    old_estimate = current_estimate

    print("Running average:", current_estimate)

    if val == -1:
        break
