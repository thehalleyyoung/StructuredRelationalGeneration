
def mod12Same(xs, ys):
    # Check if both lists are of the same length and
    # corresponding elements have a modulo 12 difference of less than 2
    return len(xs) == len(ys) and all([
        abs((xs[i] % 12) - (ys[i] % 12)) < 2 for i in range(len(xs))
    ])

def veryDifferentLengths(xs, ys):
    # Determine if the lengths of two lists differ significantly by
    # checking if the absolute difference in lengths is 4 or more, or
    # if one list is at least 3 times longer than the other
    return abs(len(xs) - len(ys)) >= 4 or len(xs) / len(ys) >= 3 or len(ys) / len(xs) >= 3

def listSequenceIn(a, b):
    # Check if list 'a' is a subsequence of list 'b'
    return any(map(lambda x: b[x:x + len(a)] == a, range(len(b) - len(a) + 1)))
