def ext_gcd(a, b):
    '''
    Extended Euclidean Algorithm
    x1 = y2
    y1 = x2 - (a//b)*y2
    xn, yn would be 1, 0 or 0, 1
    return x1 and y1
    '''
    if b == 0:
        return 1, 0
    elif a % b == 0:
        return 0, 1
    else:
        x, y = ext_gcd(b, a%b)
        return y, x - (a//b)*y