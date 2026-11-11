from src.soil_water_balance import (
    SoilConfig,
    SoilWaterBalance
)

soil = SoilWaterBalance(SoilConfig())

print("SOIL-WATER BALANCE TEST")
print("=" * 45)

initial_moisture = soil.get_soil_moisture()

print(f"Initial moisture: {initial_moisture:.3f}")

# Day 1: No rainfall or irrigation
day1 = soil.step(
    rainfall=0,
    irrigation=0,
    et0=5.0
)

print("\nDAY 1 - Dry Weather")
print(day1)

assert day1["soil_moisture"] < initial_moisture

# Day 2: Irrigation
day2 = soil.step(
    rainfall=0,
    irrigation=8.0,
    et0=5.0
)

print("\nDAY 2 - Irrigation")
print(day2)

assert day2["soil_moisture"] > day1["soil_moisture"]

# Day 3: Heavy rainfall
day3 = soil.step(
    rainfall=60.0,
    irrigation=0,
    et0=4.0
)

print("\nDAY 3 - Heavy Rainfall")
print(day3)

assert day3["drainage_mm"] > 0

print("\nAll basic soil-water tests PASSED!")