import math

def saturation_vapor_pressure(temp_c):
    """Saturation vapor pressure in kPa."""
    return 0.6108 * math.exp(
        (17.27 * temp_c) / (temp_c + 237.3)
    )

def wind_speed_2m(wind_speed_10m):
    """Convert wind speed from 10 m to 2 m."""
    return wind_speed_10m * (
        4.87 / math.log(67.8 * 10 - 5.42)
    )

def calculate_et0(
    t_min,
    t_max,
    t_mean,
    relative_humidity,
    solar_radiation,
    wind_speed_10m,
    latitude=42.8746,
    elevation=800.0,
    day_of_year=180
):
    """
    Daily FAO-56 Penman-Monteith reference ET0.

    solar_radiation: MJ/m²/day
    wind_speed_10m: m/s
    elevation: meters above sea level
    Returns: ET0 in mm/day

    elevation=800 m is a provisional study assumption.
    """

    if not (-90 <= latitude <= 90):
        raise ValueError("Invalid latitude")
    if not (1 <= day_of_year <= 366):
        raise ValueError("Invalid day of year")
    if not (0 <= relative_humidity <= 100):
        raise ValueError("Invalid relative humidity")
    if solar_radiation < 0 or wind_speed_10m < 0:
        raise ValueError("Radiation and wind must be nonnegative")
    if t_min > t_max:
        raise ValueError("Minimum temperature exceeds maximum")

    # 1. Saturation vapor pressure
    es = (
        saturation_vapor_pressure(t_max)
        + saturation_vapor_pressure(t_min)
    ) / 2

    # Approximation using mean relative humidity
    ea = (
        saturation_vapor_pressure(t_mean)
        * relative_humidity / 100
    )

    # 2. Slope of saturation vapor pressure curve
    delta = (
        4098 * saturation_vapor_pressure(t_mean)
        / ((t_mean + 237.3) ** 2)
    )

    # 3. Atmospheric pressure and psychrometric constant
    pressure = 101.3 * (
        (293 - 0.0065 * elevation) / 293
    ) ** 5.26

    gamma = 0.000665 * pressure

    # 4. Wind speed at 2 m
    u2 = wind_speed_2m(wind_speed_10m)

    # 5. Extraterrestrial radiation
    phi = math.radians(latitude)
    j = day_of_year

    dr = 1 + 0.033 * math.cos(2 * math.pi * j / 365)
    solar_declination = (
        0.409 * math.sin(2 * math.pi * j / 365 - 1.39)
    )

    sunset_angle = math.acos(
        max(-1, min(1, -math.tan(phi) * math.tan(solar_declination)))
    )

    ra = (
        (24 * 60 / math.pi)
        * 0.0820
        * dr
        * (
            sunset_angle * math.sin(phi)
            * math.sin(solar_declination)
            + math.cos(phi)
            * math.cos(solar_declination)
            * math.sin(sunset_angle)
        )
    )

    # 6. Clear-sky radiation
    rso = (0.75 + 2e-5 * elevation) * ra

    # 7. Net shortwave radiation
    rns = (1 - 0.23) * solar_radiation

    # 8. Net longwave radiation
    sigma = 4.903e-9
    tmax_k = t_max + 273.16
    tmin_k = t_min + 273.16

    radiation_ratio = min(
        1.0,
        max(0.3, solar_radiation / max(rso, 1e-9))
    )

    rnl = (
        sigma
        * ((tmax_k ** 4 + tmin_k ** 4) / 2)
        * (0.34 - 0.14 * math.sqrt(max(ea, 0)))
        * (1.35 * radiation_ratio - 0.35)
    )

    rn = rns - rnl

    # Daily soil heat flux approximation
    g = 0.0

    # 9. FAO-56 Penman-Monteith
    numerator = (
        0.408 * delta * (rn - g)
        + gamma
        * (900 / (t_mean + 273))
        * u2
        * (es - ea)
    )

    denominator = delta + gamma * (1 + 0.34 * u2)

    et0 = numerator / denominator

    return max(0.0, et0)