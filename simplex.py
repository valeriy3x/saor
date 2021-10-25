import numpy as np
from copy import copy

import scipy.linalg
from scipy.optimize import linprog


def inverse_matrix(A_1, x, i):
    n = len(x)
    l = A_1.dot(x)

    if l[i] == 0:
        return False, []

    l_1 = copy(l)
    l_1[i] = -1
    l_2 = l_1 / (-l[i])

    Q = np.eye(n)
    Q[:, i] = l_2
    R = np.zeros((n, n))
    for j in range(n):
        for k in range(n):
            if j == i:
                R[j, k] = Q[j, j] * A_1[j, k]
            else:
                R[j, k] = A_1[j, k] + Q[j, i] * A_1[i, k]

    return True, R


def simplex_main(A, c, x, J_b):
    x = copy(x)
    last_inv = None
    iteration = 1

    while True:
        m = len(J_b)

        # 1 and 2
        A_b = np.zeros((m, m))
        c_b = []

        for i, j in enumerate(J_b):
            A_b[:, i] = A[:, j]
            c_b.append(c[j])

        # lab 1 usage
        if last_inv is not None:
            A_b1 = inverse_matrix(last_inv, A[:, j_0], theta.index(min_theta))[1]
        else:
            A_b1 = np.linalg.inv(A_b)

        last_inv = A_b1

        c_b = np.array(c_b)
        u = c_b.dot(A_b1)

        # 3
        delta = u.dot(A) - c
        J_n = [i for i, j in enumerate(delta) if i not in J_b and j < 0]

        if len(J_n) == 0:
            return x, sorted(J_b)

        # 4
        j_0 = J_n[0]
        z = A_b1.dot(A[:, j_0])

        # 5
        theta = []
        for i in range(m):
            if z[i] > 0:
                theta.append(x[J_b[i]] / z[i])
            else:
                theta.append(np.inf)

        min_theta = min(theta)
        if min_theta == np.inf:
            return "Not limited"

        # 6
        J_b[theta.index(min_theta)] = j_0
        x[j_0] = min_theta

        for i, j in enumerate(J_b):
            if i != theta.index(min_theta):
                x[j] = x[j] - min_theta * z[i]

        for j in range(len(x)):
            if j not in J_b:
                x[j] = 0

        iteration += 1


def simplex_first(A, b):
    m, n = A.shape

    for i in range(m):
        if b[i] < 0:
            b[i] *= -1
            for j in range(n):
                A[i][j] *= -1

    x = [0] * n + b
    c = [0] * n + [-1] * m

    A_hat = np.hstack((A, np.eye(m)))
    J_b = [n + i for i in range(m)]
    x0, J_b0 = simplex_main(A_hat, c, x, J_b)
    for i in range(m):
        if x0[n + i] != 0:
            print("No solution")
            return None, None

    while True:
        i = -1
        bad_j = -1

        J_b_len = len(J_b0)
        for j in range(J_b_len):
            if J_b0[j] > n - 1:
                i = J_b0[j] - n + 1
                bad_j = j
                break

        if i == -1:
            # print("Bounded")
            # print(A_hat[:n])
            return x0[:n], J_b0

        Ab = np.zeros((J_b_len, J_b_len))
        for k in range(J_b_len):
            Ab[:, k] = A_hat[:, J_b0[k]]

        is_end = False
        for j in range(n):
            if j not in J_b0:
                if np.linalg.det(Ab) == 0:
                    break
                l_j = (np.linalg.inv(Ab)).dot(A_hat[:, j])
                # print("l_j: ", l_j)
                if l_j[bad_j] != 0:
                    J_b0[bad_j] = j
                    is_end = True
                    break

        if is_end:
            continue

        del J_b0[bad_j]
        A_hat = np.delete(A_hat, bad_j, axis=0)
        A = A_hat[:, :n]

def solve(A, b, c):
    x0, indices = simplex_first(A, b)
    return x0, indices
