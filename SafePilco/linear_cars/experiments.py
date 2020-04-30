from safe_cars_run import safe_cars


name = "results/"
for i in range(10):
    safe_cars(name=name+str(i), seed=i, logging=True)
