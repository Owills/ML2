def second_largest(l):
    unique_count = len(set(l))
    if unique_count < 2:
        return "none"
    l.sort()
    return l[-2]

print(second_largest([23, 52, 11, 52, 9, 26, 2, 1, 67]))
print(second_largest([12, 45, 2, 41, 31, 10, 8, 6, 4]))
print(second_largest([23,23]))
print(second_largest([24]))