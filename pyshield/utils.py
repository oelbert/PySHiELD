import datetime as dt

def date_to_julian(iyear: int, imonth: int, iday: int) -> int:
    """
    Converts integer day, month, and year to julian day number.
    Fortran name is iw3jdn:
    Computes julian day number from year (4 digits), month,
    and day. iw3jdn is valid for years 1583 a.d. to 3300 a.d.
    Julian day number can be used to compute day of week, day of
    year, record numbers in an archive, replace day of century,
    find the number of days between two dates.
    Program history log:
    - Ralph Jones 1987-03-29
    - Ralph Jones 1989-10-25 Convert to cray cft77 fortran.
    @param[in] IYEAR Integer year (4 Digits)
    @param[in] MONTH Integer month of year (1 - 12)
    @param[in] IDAY Integer day of month (1 - 31)
    @return IW3JDN Integer Julian day number
    - Jan 1, 1960 is Julian day number 2436935
    - Jan 1, 1987 is Julian day number 2446797

    Args:
        iday (int): day of month (1-31)
        imonth (int): month number (1-12)
        iyear (int): 4-digit year

    Returns:
        jdn (int): julian day number, i.e. days since 01/01 4713 BC
    """
    jdn = int(
        iday
        - 32075
        + 1461 * (iyear + 4800 + (imonth - 14) / 12) / 4
        + 367 * (imonth - 2 - (imonth - 14) / 12 * 12) / 12
        - 3 * ((iyear + 4900 + (imonth - 14) / 12) / 100) / 4
    )
    return jdn

def datetime_to_julian(datetime: dt.datetime) -> int:
    return date_to_julian(datetime.year, datetime.month, datetime.day)
