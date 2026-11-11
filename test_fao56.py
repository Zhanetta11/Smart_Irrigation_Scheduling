from src.fao56 import calculate_et0

et0 = calculate_et0(
    t_min=15,
    t_max=30,
    t_mean=22.5,
    relative_humidity=50,
    solar_radiation=22,
    wind_speed_10m=2.5,
    latitude=42.8746,
    elevation=800,
    day_of_year=180
)

print("FAO-56 PENMAN-MONTEITH")
print("=" * 40)
print(f"Reference ET0: {et0:.2f} mm/day")

assert 0 <= et0 <= 15
print("Basic plausibility check: PASSED")