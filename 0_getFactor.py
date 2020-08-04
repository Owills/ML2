def get_factor(num):
    print()
    if num < 0:
        print('Error: number is negative')
        return
    if num == 0:
        print('All numbers are factors of 0')
        return
    for x in range(1, num+1):
        if num % x == 0:
            print(x, ' is a factor of ', num)

get_factor(21)