from functools import reduce

zeroes_matrix = lambda rows, cols: [[0 for _ in range(cols)] for _ in range(rows)]
pretty_print_matrix = lambda x: print("\n".join(map(lambda y: f"{y}", x)))
multiply_elements = lambda lst: reduce(lambda x, y: x * y, lst)
sum_elements = lambda lst: reduce(lambda x, y: x + y, lst)


def transpose_matrix(matrix):
    if not isinstance(matrix[0], list):
        matrix = [matrix]
    return list(map(list, zip(*matrix)))


def multiply_matrices(left, right):
    if not isinstance(left[0], list):
        left = [left]
    if not isinstance(right[0], list):
        right = [right]
    rowsLeft, colsLeft, rowsRight, colsRight = (
        len(left),
        len(left[0]),
        len(right),
        len(right[0]),
    )
    if colsLeft != rowsRight:
        raise ValueError(
            "Number of columns in left matrix must equal number of rows in right matrix"
        )
    right_transposed = transpose_matrix(right)
    result = zeroes_matrix(rowsLeft, colsRight)
    for n_row, row in enumerate(left):
        for n_column, row_right in enumerate(right_transposed):
            result[n_row][n_column] = sum_elements(
                [multiply_elements(row) for row in zip(row, row_right)]
            )
    if rowsLeft == 1 and colsRight == 1:
        return result[0][0]
    return result


add_vectors = lambda left, right: list(map(lambda x: x[0] + x[1], zip(left, right)))
subtract_vectors = lambda left, right: list(
    map(lambda x: x[0] - x[1], zip(left, right))
)


class DimensionMismatchException(Exception):
    def __init__(self, message="Matrices must have the same dimensions"):
        self.message = message
        super().__init__(self.message)


def add_matrices(left, right):
    if (len(left) != len(right)) or (len(left[0]) != len(right[0])):
        raise DimensionMismatchException
    return list(map(lambda x: add_vectors(x[0], x[1]), zip(left, right)))


def subtract_matrices(left, right):
    if (len(left) != len(right)) or (len(left[0]) != len(right[0])):
        raise DimensionMismatchException
    return list(map(lambda x: subtract_vectors(x[0], x[1]), zip(left, right)))


def multiply_matrix_by_scalar(scalar, matrix):
    return list(map(lambda x: list(map(lambda y: scalar * y, x)), matrix))


def get_minor(m, i, j):
    return [row[:j] + row[j + 1 :] for row in (m[:i] + m[i + 1 :])]


def get_determinant(m):
    # base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1) ** c) * m[0][c] * get_determinant(get_minor(m, 0, c))
    return determinant


def get_matrix_inverse(m):
    determinant = get_determinant(m)
    # special case for 2x2 matrix:
    if len(m) == 2:
        return [
            [m[1][1] / determinant, -1 * m[0][1] / determinant],
            [-1 * m[1][0] / determinant, m[0][0] / determinant],
        ]

    # find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = get_minor(m, r, c)
            cofactorRow.append(((-1) ** (r + c)) * get_determinant(minor))
        cofactors.append(cofactorRow)
    cofactors = transpose_matrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c] / determinant
    return cofactors


def linear_projection(points, slope, intercept):
    projected_values = []

    for point in points:
        x, y = point
        projected_y = slope * x + intercept
        projected_values.append((x, projected_y))

    return projected_values
