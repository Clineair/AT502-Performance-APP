import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Constants - Air Tractor AT-502B base data
# =============================================================================

# Base performance (sea level, ISA 15°C, no wind, max takeoff 9400 lbs)
BASE_TAKEOFF_GROUND_ROLL_FT    = 1140
BASE_TAKEOFF_TO_50FT_FT        = 2600
BASE_LANDING_GROUND_ROLL_FT    = 600
BASE_LANDING_TO_50FT_FT        = 1350
BASE_CLIMB_RATE_FPM            = 870
BASE_STALL_FLAPS_DOWN_MPH      = 68      # at 8000 lbs
BEST_CLIMB_SPEED_MPH           = 111

BASE_EMPTY_WEIGHT_LBS          = 4546
BASE_FUEL_CAPACITY_GAL         = 170
FUEL_WEIGHT_PER_GAL            = 6.0
HOPPER_CAPACITY_GAL            = 500
HOPPER_WEIGHT_PER_GAL          = 8.0      # chemical/water approx
MAX_TAKEOFF_WEIGHT_LBS         = 9400
MAX_LANDING_WEIGHT_LBS         = 8000

GLIDE_RATIO                    = 8.0      # approx clean

# Runway condition performance factors (multipliers)
# These are approximate / conservative values – adjust based on real data/POH
RUNWAY_CONDITION_FACTORS = {
    "Dry hard surface":         1.00,
    "Wet hard surface":         1.25,
    "Standing water":           1.60,
    "Grass":                    1.70,
    "Dirt or gravel":           1.15,
    "Rough with potholes":      1.55,
    "Poor surface":             1.40,
}

# =============================================================================
# Helper Functions
# =============================================================================

def calculate_density_altitude(pressure_alt_ft, oat_c):
    isa_temp_c = 15 - (2 * pressure_alt_ft / 1000)
    da = pressure_alt_ft + (120 * (oat_c - isa_temp_c))
    return da


def adjust_for_weight(value, current_weight, base_weight, exponent=1.5):
    return value * (current_weight / base_weight) ** exponent


def adjust_for_wind(value, wind_kts):
    # headwind positive → reduces distance
    factor = 1 - (0.1 * wind_kts / 9)
    return value * max(factor, 0.5)  # floor at 50%


def adjust_for_da(value, da_ft):
    factor = 1 + (0.07 * da_ft / 1000)  # ~7% per 1000 ft DA
    return value * factor


def adjust_for_runway_condition(value, condition):
    factor = RUNWAY_CONDITION_FACTORS.get(condition, 1.40)
    return value * factor


# =============================================================================
# Performance Calculations (cached)
# =============================================================================

@st.cache_data
def compute_takeoff(pressure_alt_ft, oat_c, weight_lbs, wind_kts, runway_condition):
    da_ft = calculate_density_altitude(pressure_alt_ft, oat_c)

    ground_roll = adjust_for_weight(BASE_TAKEOFF_GROUND_ROLL_FT, weight_lbs, MAX_TAKEOFF_WEIGHT_LBS)
    ground_roll = adjust_for_da(ground_roll, da_ft)
    ground_roll = adjust_for_wind(ground_roll, wind_kts)
    ground_roll = adjust_for_runway_condition(ground_roll, runway_condition)

    to_50ft = adjust_for_weight(BASE_TAKEOFF_TO_50FT_FT, weight_lbs, MAX_TAKEOFF_WEIGHT_LBS)
    to_50ft = adjust_for_da(to_50ft, da_ft)
    to_50ft = adjust_for_wind(to_50ft, wind_kts)
    to_50ft = adjust_for_runway_condition(to_50ft, runway_condition)

    return ground_roll, to_50ft


@st.cache_data
def compute_landing(pressure_alt_ft, oat_c, weight_lbs, wind_kts, runway_condition):
    weight_lbs = min(weight_lbs, MAX_LANDING_WEIGHT_LBS)

    da_ft = calculate_density_altitude(pressure_alt_ft, oat_c)

    ground_roll = adjust_for_weight(BASE_LANDING_GROUND_ROLL_FT, weight_lbs, MAX_LANDING_WEIGHT_LBS, exponent=1.0)
    ground_roll = adjust_for_da(ground_roll, da_ft)
    ground_roll = adjust_for_wind(ground_roll, wind_kts)
    ground_roll = adjust_for_runway_condition(ground_roll, runway_condition)

    from_50ft = adjust_for_weight(BASE_LANDING_TO_50FT_FT, weight_lbs, MAX_LANDING_WEIGHT_LBS, exponent=1.0)
    from_50ft = adjust_for_da(from_50ft, da_ft)
    from_50ft = adjust_for_wind(from_50ft, wind_kts)
    from_50ft = adjust_for_runway_condition(from_50ft, runway_condition)

    return ground_roll, from_50ft


@st.cache_data
def compute_climb_rate(pressure_alt_ft, oat_c, weight_lbs):
    da_ft = calculate_density_altitude(pressure_alt_ft, oat_c)
    climb = adjust_for_weight(BASE_CLIMB_RATE_FPM, weight_lbs, MAX_TAKEOFF_WEIGHT_LBS, exponent=-1)
    climb *= (1 - (0.05 * da_ft / 1000))
    return max(climb, 0)


@st.cache_data
def compute_stall_speed(weight_lbs):
    return BASE_STALL_FLAPS_DOWN_MPH * np.sqrt(weight_lbs / MAX_LANDING_WEIGHT_LBS)


@st.cache_data
def compute_glide_distance(height_ft, wind_kts):
    ground_speed_mph = 100 + wind_kts
    glide_distance_nm = (height_ft / 6076) * GLIDE_RATIO * (ground_speed_mph / 60)
    return glide_distance_nm


@st.cache_data
def compute_weight_balance(empty_weight_lbs, fuel_gal, hopper_gal, pilot_weight_lbs):
    fuel_weight   = fuel_gal   * 6.0
    hopper_weight = hopper_gal * 8.0
    total_weight  = empty_weight_lbs + fuel_weight + hopper_weight + pilot_weight_lbs

    status = "Within limits" if total_weight <= MAX_TAKEOFF_WEIGHT_LBS else "Overweight!"
    if total_weight > MAX_LANDING_WEIGHT_LBS:
        status += " (Exceeds max landing weight)"

    return total_weight, status

# =============================================================================
# Streamlit App
# =============================================================================

st.set_page_config(page_title="AT-502B Performance Calculator", layout="wide")

st.title("Air Tractor AT-502B Performance Calculator")
st.caption("Prototype – based on public data. Not for flight planning. Always use official POH / AFM.")

# ── INPUTS ────────────────────────────────────────────────────────────────

st.subheader("Input Conditions")

col1, col2 = st.columns(2)

with col1:
    pressure_alt_ft = st.number_input("Pressure Altitude (ft)", 0, 20000, 0, step=100)
    oat_c           = st.number_input("OAT (°C)", -30, 50, 15, step=1)
    weight_lbs      = st.number_input("Gross Weight (lbs)", 4000, 9400, 9400, step=50)
    wind_kts        = st.number_input("Headwind (+) / Tailwind (-) (kts)", -20, 20, 0, step=1)

with col2:
    fuel_gal        = st.number_input("Fuel (gal)", 0, 170, 170, step=10)
    hopper_gal      = st.number_input("Hopper Load (gal)", 0, 500, 0, step=50)
    pilot_weight_lbs = st.number_input("Pilot Weight (lbs)", 100, 300, 200, step=10)
    glide_height_ft = st.number_input("Glide Height AGL (ft)", 0, 15000, 1000, step=100)

    runway_condition = st.selectbox(
        "Runway Surface Condition",
        options=list(RUNWAY_CONDITION_FACTORS.keys()),
        index=0,
        help="Affects takeoff and landing distances. Conservative estimates used."
    )

# ── CALCULATE BUTTON & RESULTS ───────────────────────────────────────────

if st.button("Calculate Performance", type="primary"):

    # Calculations
    ground_roll_to, to_50ft     = compute_takeoff(pressure_alt_ft, oat_c, weight_lbs, wind_kts, runway_condition)
    ground_roll_land, from_50ft = compute_landing(pressure_alt_ft, oat_c, weight_lbs, wind_kts, runway_condition)
    climb_rate                  = compute_climb_rate(pressure_alt_ft, oat_c, weight_lbs)
    stall_speed                 = compute_stall_speed(weight_lbs)
    glide_dist                  = compute_glide_distance(glide_height_ft, wind_kts)
    total_weight, cg_status     = compute_weight_balance(4546, fuel_gal, hopper_gal, pilot_weight_lbs)

    # ── RESULTS ───────────────────────────────────────────────────────────
    st.subheader("Performance Results")

    col_a, col_b = st.columns(2)

    with col_a:
        st.metric("Takeoff Ground Roll", f"{ground_roll_to:.0f} ft")
        st.metric("Takeoff to 50 ft", f"{to_50ft:.0f} ft")
        st.metric("Landing Ground Roll", f"{ground_roll_land:.0f} ft")
        st.metric("Landing from 50 ft", f"{from_50ft:.0f} ft")

    with col_b:
        st.metric("Climb Rate", f"{climb_rate:.0f} fpm")
        st.metric("Best Rate Climb Speed", f"{BEST_CLIMB_SPEED_MPH} mph IAS")
        st.metric("Stall Speed (flaps down)", f"{stall_speed:.1f} mph")
        st.metric("Emergency Glide Distance", f"{glide_dist:.1f} nm")

    st.markdown(f"**Total Weight:** {total_weight:.0f} lbs  –  **{cg_status}**")

    # Runway condition explanation
    factor = RUNWAY_CONDITION_FACTORS.get(runway_condition, 1.4)
    st.info(f"**Runway condition impact:** {runway_condition} → ×{factor:.2f} on takeoff & landing distances")

    # ── CLIMB CHART ───────────────────────────────────────────────────────
    st.subheader("Rate of Climb vs Pressure Altitude")
    altitudes = np.linspace(0, 12000, 60)
    climb_rates = [compute_climb_rate(alt, oat_c, weight_lbs) for alt in altitudes]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(altitudes, climb_rates, color='darkgreen', linewidth=2.2, marker='o', markersize=4)
    ax.set_xlabel("Pressure Altitude (ft)")
    ax.set_ylabel("Rate of Climb (fpm)")
    ax.set_title(f"Climb Performance – OAT {oat_c}°C, Weight {weight_lbs} lbs")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.minorticks_on()
    st.pyplot(fig)
st.markdown("---")
# ────────────────────────────────────────────────
# Feedback – Star Rating + Comment Box
# ────────────────────────────────────────────────

st.markdown("---")
st.subheader("Rate this AT-502B Calculator")

# Define the star rating widget FIRST
rating = st.feedback("stars")

# Comment box (always visible)
comment = st.text_area(
    "Comments, suggestions or issues",
    height=120,
    placeholder="What would make this tool more useful? What would you like to see? Questions about the output?..."
)

# Submit logic – now safe because rating is always defined
if st.button("Submit Rating & Comment"):
    if rating is not None:
        stars = rating + 1
        st.success(f"Thank you! You rated **{stars} stars**")
        if comment.strip():
            st.caption(f"Comment: {comment}")
        else:
            st.caption("No comment provided.")
    else:
        st.warning("Please select a star rating before submitting.")

