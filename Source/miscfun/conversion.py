def db2pow(x):
    y = 10 ** (x / 10)
    return (y)


def search_max(x, kmax):
    m = []
    value = []
    for k in range(0, kmax):
        v = np.argmax(x)
        value = np.append(value, x[v])
        x[v] = -np.Inf
        m = np.append(m, int(v))

    return np.array(m), np.array(value)
