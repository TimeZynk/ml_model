def top_n(prob, n):
    res = []
    for index, value in enumerate(prob):
        if len(res) < n:
            res.append((index, value))
        elif len(res) == n and value > res[-1][1]:
            res[-1] = (index, value)
        res.sort(reverse=True, key=lambda x: x[1])
    return list(map(lambda x: x[0], res))
