import random
import numpy as np


def estimate_price(num_samples):
    off_p = 0
    day_time = 0
    peak = 0
    x_off = []
    x_day = []
    x_peak = []
    x_all = []

    for i in range(num_samples):
        x = random.uniform(30, 112)

        x_all.append(x)

        if ((x >= 30) == True) & ((x < 50) == True):
            off_p += 1
            x_off.append(x)

        else:
            if ((x >= 50) == True) & ((x < 87) == True):
                day_time += 1
                x_day.append(x)
            else:
                if x > 70:
                    peak += 1
                    x_peak.append(x)
    print(x_all)
    print(x_day)
    print(x_off)
    print(x_peak)
    print(off_p)
    print(day_time)
    print(peak)
    n_all = num_samples
    return (
        np.mean(x_all),
        np.mean(x_off),
        np.mean(x_day),
        np.mean(x_peak),
        off_p / n_all,
        day_time / n_all,
        peak / n_all,
    )
