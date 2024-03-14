from gt4py.cartesian.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval

from ndsl.dsl.typing import FloatField


def tridit(
    au: FloatField,
    cm: FloatField,
    cl: FloatField,
    f1: FloatField,
):
    with computation(FORWARD):
        with interval(0, 1):
            fk = 1.0 / cm[0, 0, 0]
            au = fk * au[0, 0, 0]
            f1 = fk * f1[0, 0, 0]
        with interval(1, -1):
            fkk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
            au = fkk * au[0, 0, 0]
            f1 = fkk * (f1[0, 0, 0] - cl[0, 0, -1] * f1[0, 0, -1])

    with computation(BACKWARD):
        with interval(-1, None):
            fk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
            f1 = fk * (f1[0, 0, 0] - cl[0, 0, -1] * f1[0, 0, -1])
        with interval(0, -1):
            f1 = f1[0, 0, 0] - au[0, 0, 0] * f1[0, 0, 1]


def tridi2(
    a1: FloatField,
    a2: FloatField,
    au: FloatField,
    cl: FloatField,
    cm: FloatField,
    cu: FloatField,
    r1: FloatField,
    r2: FloatField,
):

    with computation(PARALLEL), interval(0, 1):
        fk = 1 / cm[0, 0, 0]
        au = fk * cu[0, 0, 0]
        a1 = fk * r1[0, 0, 0]
        a2[0, 0, 0][0] = fk * r2[0, 0, 0][0]

    with computation(FORWARD):
        with interval(1, -1):
            fk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
            au = fk * cu[0, 0, 0]
            a1 = fk * (r1[0, 0, 0] - cl[0, 0, -1] * a1[0, 0, -1])
            a2[0, 0, 0][0] = fk * (r2[0, 0, 0][0] - cl[0, 0, -1] * a2[0, 0, -1][0])
        with interval(-1, None):
            fk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
            a1 = fk * (r1[0, 0, 0] - cl[0, 0, -1] * a1[0, 0, -1])
            a2[0, 0, 0][0] = fk * (r2[0, 0, 0][0] - cl[0, 0, -1] * a2[0, 0, -1][0])

    with computation(BACKWARD), interval(0, -1):
        a1 = a1[0, 0, 0] - au[0, 0, 0] * a1[0, 0, 1]
        a2[0, 0, 0][0] = a2[0, 0, 0][0] - au[0, 0, 0] * a2[0, 0, 1][0]


def tridin(
    cl: FloatField,
    cm: FloatField,
    cu: FloatField,
    r1: FloatField,
    r2: FloatField,
    au: FloatField,
    a1: FloatField,
    a2: FloatField,
):
    from __externals__ import nt

    with computation(FORWARD):
        with interval(0, 1):
            fk = 1.0 / cm[0, 0, 0]
            au = fk * cu[0, 0, 0]
            a1 = fk * r1[0, 0, 0]
            for n0 in range(nt):
                a2[0, 0, 0][n0] = fk * r2[0, 0, 0][n0]

        with interval(1, -1):
            fkk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
            au = fkk * cu[0, 0, 0]
            a1 = fkk * (r1[0, 0, 0] - cl[0, 0, -1] * a1[0, 0, -1])

            for n1 in range(nt):
                a2[0, 0, 0][n1] = fkk * (
                    r2[0, 0, 0][n1] - cl[0, 0, -1] * a2[0, 0, -1][n1]
                )

        with interval(-1, None):
            fk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
            a1 = fk * (r1[0, 0, 0] - cl[0, 0, -1] * a1[0, 0, -1])

            for n2 in range(nt):
                a2[0, 0, 0][n2] = fk * (
                    r2[0, 0, 0][n2] - cl[0, 0, -1] * a2[0, 0, -1][n2]
                )

    with computation(BACKWARD):
        with interval(0, -1):
            a1 = a1[0, 0, 0] - au[0, 0, 0] * a1[0, 0, 1]
            for n3 in range(nt):
                a2[0, 0, 0][n3] = a2[0, 0, 0][n3] - au[0, 0, 0] * a2[0, 0, 1][n3]
