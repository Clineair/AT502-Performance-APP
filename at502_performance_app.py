# Add near RATINGS_FILE
RUNWAY_FILE = "runway_conditions.json"

RUNWAY_CONDITIONS = [
    "Dry",
    "Wet",
    "Wet with visible moisture",
    "Standing water",
    "Snow",
    "Compacted snow",
    "Slush",
    "Ice",
    "Frost",
    "Contaminated (mix)",
    "Not reported / Unknown"
]

# Add these two functions (you can put them near your rating functions)
def load_json(file_path, default=None):
    if default is None:
        default = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except:
            return default
    return default

def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

def get_saved_runway_condition(item_id):
    data = load_json(RUNWAY_FILE)
    return data.get(item_id)

def save_runway_condition(item_id, condition):
    data = load_json(RUNWAY_FILE)
    data[item_id] = condition
    save_json(RUNWAY_FILE, data)

import streamlit as st

import numpy as np

import matplotlib.pyplot as plt



# Base data from Air Tractor AT-502B specs (sea level, ISA 15°C, no wind, max takeoff weight 9,400 lbs, max landing 8,000 lbs)

BASE_TAKEOFF_GROUND_ROLL_FT = 1140  # Ground roll to liftoff

BASE_TAKEOFF_TO_50FT_FT = 2600  # Approximate to 50 ft obstacle

BASE_LANDING_GROUND_ROLL_FT = 600  # Approximate ground roll

BASE_LANDING_TO_50FT_FT = 1350  # Approximate from 50 ft obstacle

BASE_CLIMB_RATE_FPM = 870

BASE_STALL_FLAPS_DOWN_MPH = 68  # At 8,000 lbs; adjust for weight

BEST_CLIMB_SPEED_MPH = 111  # Best rate IAS from training data

BASE_EMPTY_WEIGHT_LBS = 4546

BASE_FUEL_CAPACITY_GAL = 170

FUEL_WEIGHT_PER_GAL = 6  # Avg avgas/jet fuel density lbs/gal

HOPPER_CAPACITY_GAL = 500

HOPPER_WEIGHT_PER_GAL = 8  # Assume water/chemical density lbs/gal

MAX_TAKEOFF_WEIGHT_LBS = 9400

MAX_LANDING_WEIGHT_LBS = 8000

GLIDE_RATIO = 8  # Approximate clean glide ratio (distance/height)



def calculate_density_altitude(pressure_alt_ft, oat_c):

    isa_temp_c = 15 - (2 * pressure_alt_ft / 1000)

    da = pressure_alt_ft + (120 * (oat_c - isa_temp_c))

    return da



def adjust_for_weight(value, current_weight, base_weight, exponent=1.5):

    return value * (current_weight / base_weight) ** exponent



def adjust_for_wind(value, wind_kts):

    # Positive wind_kts = headwind (reduces distance), negative = tailwind

    factor = 1 - (0.1 * wind_kts / 9)  # 10% change per 9 kts headwind

    return value * max(factor, 0.5)  # Prevent negative/too low



def adjust_for_da(value, da_ft):

    factor = 1 + (0.07 * da_ft / 1000)  # 7% increase per 1,000 ft DA

    return value * factor



@st.cache_data

def compute_takeoff(pressure_alt_ft, oat_c, weight_lbs, wind_kts):

    da_ft = calculate_density_altitude(pressure_alt_ft, oat_c)

    ground_roll = adjust_for_weight(BASE_TAKEOFF_GROUND_ROLL_FT, weight_lbs, MAX_TAKEOFF_WEIGHT_LBS)

    ground_roll = adjust_for_da(ground_roll, da_ft)

    ground_roll = adjust_for_wind(ground_roll, wind_kts)

    

    to_50ft = adjust_for_weight(BASE_TAKEOFF_TO_50FT_FT, weight_lbs, MAX_TAKEOFF_WEIGHT_LBS)

    to_50ft = adjust_for_da(to_50ft, da_ft)

    to_50ft = adjust_for_wind(to_50ft, wind_kts)

    

    return ground_roll, to_50ft



@st.cache_data

def compute_landing(pressure_alt_ft, oat_c, weight_lbs, wind_kts):

    weight_lbs = min(weight_lbs, MAX_LANDING_WEIGHT_LBS)  # Cap at max landing weight

    da_ft = calculate_density_altitude(pressure_alt_ft, oat_c)

    ground_roll = adjust_for_weight(BASE_LANDING_GROUND_ROLL_FT, weight_lbs, MAX_LANDING_WEIGHT_LBS, exponent=1.0)  # Linear for landing

    ground_roll = adjust_for_da(ground_roll, da_ft)

    ground_roll = adjust_for_wind(ground_roll, wind_kts)

    

    from_50ft = adjust_for_weight(BASE_LANDING_TO_50FT_FT, weight_lbs, MAX_LANDING_WEIGHT_LBS, exponent=1.0)

    from_50ft = adjust_for_da(from_50ft, da_ft)

    from_50ft = adjust_for_wind(from_50ft, wind_kts)

    

    return ground_roll, from_50ft



@st.cache_data

def compute_climb_rate(pressure_alt_ft, oat_c, weight_lbs):

    da_ft = calculate_density_altitude(pressure_alt_ft, oat_c)

    climb = adjust_for_weight(BASE_CLIMB_RATE_FPM, weight_lbs, MAX_TAKEOFF_WEIGHT_LBS, exponent=-1)  # Climb decreases with weight

    climb *= (1 - (0.05 * da_ft / 1000))  # Approximate 5% loss per 1,000 ft DA

    return max(climb, 0)



@st.cache_data

def compute_stall_speed(weight_lbs):

    stall = BASE_STALL_FLAPS_DOWN_MPH * np.sqrt(weight_lbs / MAX_LANDING_WEIGHT_LBS)  # Stall ~ sqrt(weight)

    return stall



@st.cache_data

def compute_glide_distance(height_ft, wind_kts):

    ground_speed_mph = 100  # Approximate best glide speed mph

    ground_speed_mph += wind_kts  # Adjust for head/tail wind

    glide_distance_nm = (height_ft / 6076) * GLIDE_RATIO * (ground_speed_mph / 60)  # Rough calc

    return glide_distance_nm



@st.cache_data

def compute_weight_balance(empty_weight_lbs, fuel_gal, hopper_gal, pilot_weight_lbs):

    fuel_weight = fuel_gal * FUEL_WEIGHT_PER_GAL

    hopper_weight = hopper_gal * HOPPER_WEIGHT_PER_GAL

    total_weight = empty_weight_lbs + fuel_weight + hopper_weight + pilot_weight_lbs

    # Simple CG check (assume neutral arms for prototype; add real arms from POH)

    cg_status = "Within limits" if total_weight <= MAX_TAKEOFF_WEIGHT_LBS else "Overweight!"

    landing_status = "" if total_weight <= MAX_LANDING_WEIGHT_LBS else " (Exceeds max landing weight)"

    return total_weight, cg_status + landing_status



# Streamlit App UI

st.title("Air Tractor AT-502B Performance Calculator")

st.markdown("Prototype app based on public data. Not for operational use—consult official POH.")



# Inputs

col1, col2 = st.columns(2)

with col1:

    pressure_alt_ft = st.number_input("Pressure Altitude (ft)", 0, 20000, 0)

    oat_c = st.number_input("OAT (°C)", -20, 50, 15)

    weight_lbs = st.number_input("Gross Weight (lbs)", 4000, 9400, 9400)

    wind_kts = st.number_input("Headwind (+)/Tailwind (-) (kts)", -20, 20, 0)

with col2:

    fuel_gal = st.number_input("Fuel (gal)", 0, BASE_FUEL_CAPACITY_GAL, 170)

    hopper_gal = st.number_input("Hopper Load (gal)", 0, HOPPER_CAPACITY_GAL, 0)

    pilot_weight_lbs = st.number_input("Pilot Weight (lbs)", 100, 300, 200)

    glide_height_ft = st.number_input("Glide Height AGL (ft)", 0, 10000, 1000)



# Calculations

if st.button("Calculate Performance"):

    ground_roll_to, to_50ft = compute_takeoff(pressure_alt_ft, oat_c, weight_lbs, wind_kts)

    ground_roll_land, from_50ft = compute_landing(pressure_alt_ft, oat_c, weight_lbs, wind_kts)

    climb_rate = compute_climb_rate(pressure_alt_ft, oat_c, weight_lbs)

    stall_speed = compute_stall_speed(weight_lbs)

    glide_dist = compute_glide_distance(glide_height_ft, wind_kts)

    total_weight, cg_status = compute_weight_balance(BASE_EMPTY_WEIGHT_LBS, fuel_gal, hopper_gal, pilot_weight_lbs)

    

    st.subheader("Results")

    st.write(f"**Takeoff Ground Roll:** {ground_roll_to:.0f} ft")

    st.write(f"**Takeoff to 50 ft Obstacle:** {to_50ft:.0f} ft")

    st.write(f"**Landing Ground Roll:** {ground_roll_land:.0f} ft")

    st.write(f"**Landing from 50 ft Obstacle:** {from_50ft:.0f} ft")

    st.write(f"**Climb Rate (at inputs):** {climb_rate:.0f} fpm")

    st.write(f"**Best Rate Climb Speed:** {BEST_CLIMB_SPEED_MPH} mph IAS")

    st.write(f"**Stall Speed (Flaps Down):** {stall_speed:.1f} mph")

    st.write(f"**Emergency Glide Distance:** {glide_dist:.1f} nm")

    st.write(f"**Total Weight:** {total_weight:.0f} lbs ({cg_status})")

    

    # Generate Climb Performance Chart (optimized)

    st.subheader("Climb Performance Chart")

    altitudes = np.linspace(0, 10000, 50)  # More points for smoother curve

    climb_rates = [compute_climb_rate(alt, oat_c, weight_lbs) for alt in altitudes]

    

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)  # Larger size and higher DPI

    ax.plot(altitudes, climb_rates, marker='o', linestyle='-', linewidth=2)

    ax.set_xlabel('Pressure Altitude (ft)', fontsize=12)

    ax.set_ylabel('Rate of Climb (fpm)', fontsize=12)

    ax.set_title(f'Rate of Climb vs Altitude (OAT: {oat_c}°C, Weight: {weight_lbs} lbs)', fontsize=14)

    ax.grid(True, which='major', linestyle='--')

    ax.grid(True, which='minor', linestyle=':', alpha=0.5)

    ax.minorticks_on()

    st.pyplot(fig)



st.markdown("---")
