import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load the trained models
knn_model = pickle.load(open("knn_model.pkl", "rb"))
lr_model = pickle.load(open("lr_model.pkl", "rb"))
svm_model = pickle.load(open("svm_model.pkl", "rb"))


# Fitness plans dictionary
fitness_plans = {
    'Lean': {
        'Male': {
            'Goal': 'Lean',
            'Exercise Type': 'Cardio',
            'Exercise Plan': '30 mins Cardio, 3x/week Strength',
            'Cardio Minutes': 30,
            'Strength Training': '3x/week',
            'Water Intake (Liters)': 2.5,
            'Sleep Hours': 8,
            'Diet Plan': '2000 kcal, 120g Protein, 250g Carbs',
            'Protein (g)': 120,
            'Carbs (g)': 250,
            'Fats (g)': 50,
            'Recommend Fitness Routine': 'Morning: Cardio, Afternoon: Work, Evening: Stretching'
        },
        'Female': {
            'Goal': 'Lean',
            'Exercise Type': 'Cardio',
            'Exercise Plan': '20 mins Cardio, 2x/week Strength',
            'Cardio Minutes': 20,
            'Strength Training': '2x/week',
            'Water Intake (Liters)': 2,
            'Sleep Hours': 8,
            'Diet Plan': '1800 kcal, 90g Protein, 200g Carbs',
            'Protein (g)': 90,
            'Carbs (g)': 200,
            'Fats (g)': 40,
            'Recommend Fitness Routine': 'Morning: Cardio, Afternoon: Work, Evening: Stretching'
        }
    },
    'Athletic': {
        'Male': {
            'Goal': 'Athletic',
            'Exercise Type': 'Cardio',
            'Exercise Plan': '45 mins Cardio, 4x/week Strength',
            'Cardio Minutes': 45,
            'Strength Training': '4x/week',
            'Water Intake (Liters)': 3,
            'Sleep Hours': 8,
            'Diet Plan': '2500 kcal, 150g Protein, 300g Carbs',
            'Protein (g)': 150,
            'Carbs (g)': 300,
            'Fats (g)': 70,
            'Recommend Fitness Routine': 'Morning: Cardio, Afternoon: Work, Evening: Stretching'
        },
        'Female': {
            'Goal': 'Athletic',
            'Exercise Type': 'Cardio',
            'Exercise Plan': '30 mins Cardio, 3x/week Strength',
            'Cardio Minutes': 30,
            'Strength Training': '3x/week',
            'Water Intake (Liters)': 2.5,
            'Sleep Hours': 8,
            'Diet Plan': '2200 kcal, 120g Protein, 250g Carbs',
            'Protein (g)': 120,
            'Carbs (g)': 250,
            'Fats (g)': 60,
            'Recommend Fitness Routine': 'Morning: Cardio, Afternoon: Work, Evening: Stretching'
        }
    },
    'Bulky': {
        'Male': {
            'Goal': 'Bulky',
            'Exercise Type': 'Strength',
            'Exercise Plan': '5x/week Heavy Lifting, Focus on Power',
            'Cardio Minutes': 0,
            'Strength Training': '5x/week',
            'Water Intake (Liters)': 3.5,
            'Sleep Hours': 8,
            'Diet Plan': '3000 kcal, 180g Protein, 400g Carbs',
            'Protein (g)': 180,
            'Carbs (g)': 400,
            'Fats (g)': 90,
            'Recommend Fitness Routine': 'Morning: Strength, Afternoon: Rest, Evening: Cardio'
        },
        'Female': {
            'Goal': 'Bulky',
            'Exercise Type': 'Strength',
            'Exercise Plan': '4x/week Heavy Lifting, Focus on Power',
            'Cardio Minutes': 10,
            'Strength Training': '4x/week',
            'Water Intake (Liters)': 3,
            'Sleep Hours': 8,
            'Diet Plan': '2500 kcal, 150g Protein, 350g Carbs',
            'Protein (g)': 150,
            'Carbs (g)': 350,
            'Fats (g)': 80,
            'Recommend Fitness Routine': 'Morning: Strength, Afternoon: Rest, Evening: Cardio'
        }
    },
    'Ripped': {
        'Male': {
            'Goal': 'Ripped',
            'Exercise Type': 'HIIT',
            'Exercise Plan': '4x/week Strength, 2x/week Cardio',
            'Cardio Minutes': 30,
            'Strength Training': '4x/week',
            'Water Intake (Liters)': 3.5,
            'Sleep Hours': 8,
            'Diet Plan': '2800 kcal, 180g Protein, 350g Carbs',
            'Protein (g)': 180,
            'Carbs (g)': 350,
            'Fats (g)': 80,
            'Recommend Fitness Routine': 'Morning: Strength, Afternoon: Cardio, Evening: Rest'
        },
        'Female': {
            'Goal': 'Ripped',
            'Exercise Type': 'HIIT',
            'Exercise Plan': '3x/week Strength, 2x/week Cardio',
            'Cardio Minutes': 25,
            'Strength Training': '3x/week',
            'Water Intake (Liters)': 3,
            'Sleep Hours': 8,
            'Diet Plan': '2400 kcal, 140g Protein, 300g Carbs',
            'Protein (g)': 140,
            'Carbs (g)': 300,
            'Fats (g)': 70,
            'Recommend Fitness Routine': 'Morning: Strength, Afternoon: Cardio, Evening: Rest'
        }
    },
    'Jacked': {
        'Male': {
            'Goal': 'Jacked',
            'Exercise Type': 'HIIT',
            'Exercise Plan': '3x/week Strength, 3x/week HIIT',
            'Cardio Minutes': 45,
            'Strength Training': '3x/week',
            'Water Intake (Liters)': 4,
            'Sleep Hours': 8,
            'Diet Plan': '3200 kcal, 200g Protein, 400g Carbs',
            'Protein (g)': 200,
            'Carbs (g)': 400,
            'Fats (g)': 100,
            'Recommend Fitness Routine': 'Morning: Strength, Afternoon: Rest, Evening: Cardio'
        },
        'Female': {
            'Goal': 'Jacked',
            'Exercise Type': 'HIIT',
            'Exercise Plan': '3x/week Strength, 2x/week HIIT',
            'Cardio Minutes': 40,
            'Strength Training': '3x/week',
            'Water Intake (Liters)': 3.5,
            'Sleep Hours': 8,
            'Diet Plan': '2800 kcal, 180g Protein, 350g Carbs',
            'Protein (g)': 180,
            'Carbs (g)': 350,
            'Fats (g)': 90,
            'Recommend Fitness Routine': 'Morning: Strength, Afternoon: Rest, Evening: Cardio'
        }
    },
    'Mixed Routine': {
        'Male': {
            'Goal': 'Mixed Routine',
            'Exercise Type': 'Strength, Cardio',
            'Exercise Plan': '2x/week Strength, 3x/week Cardio',
            'Cardio Minutes': 30,
            'Strength Training': '2x/week',
            'Water Intake (Liters)': 3,
            'Sleep Hours': 8,
            'Diet Plan': '2500 kcal, 140g Protein, 300g Carbs',
            'Protein (g)': 140,
            'Carbs (g)': 300,
            'Fats (g)': 75,
            'Recommend Fitness Routine': 'Morning: Cardio, Afternoon: Work, Evening: Stretching'
        },
        'Female': {
            'Goal': 'Mixed Routine',
            'Exercise Type': 'Strength, Cardio',
            'Exercise Plan': '2x/week Strength, 3x/week Cardio',
            'Cardio Minutes': 25,
            'Strength Training': '2x/week',
            'Water Intake (Liters)': 2.5,
            'Sleep Hours': 8,
            'Diet Plan': '2200 kcal, 120g Protein, 270g Carbs',
            'Protein (g)': 120,
            'Carbs (g)': 270,
            'Fats (g)': 70,
            'Recommend Fitness Routine': 'Morning: Cardio, Afternoon: Work, Evening: Stretching'
        }
    },
    'Cut': {
        'Male': {
            'Goal': 'Cut',
            'Exercise Type': 'Cardio, Strength',
            'Exercise Plan': '3x/week Strength, 2x/week Cardio',
            'Cardio Minutes': 30,
            'Strength Training': '3x/week',
            'Water Intake (Liters)': 3,
            'Sleep Hours': 8,
            'Diet Plan': '2200 kcal, 140g Protein, 200g Carbs',
            'Protein (g)': 140,
            'Carbs (g)': 200,
            'Fats (g)': 60,
            'Recommend Fitness Routine': 'Morning: Cardio, Afternoon: Work, Evening: Stretching'
        },
        'Female': {
            'Goal': 'Cut',
            'Exercise Type': 'Cardio, Strength',
            'Exercise Plan': '3x/week Strength, 2x/week Cardio',
            'Cardio Minutes': 25,
            'Strength Training': '3x/week',
            'Water Intake (Liters)': 2.5,
            'Sleep Hours': 8,
            'Diet Plan': '2000 kcal, 120g Protein, 180g Carbs',
            'Protein (g)': 120,
            'Carbs (g)': 180,
            'Fats (g)': 55,
            'Recommend Fitness Routine': 'Morning: Cardio, Afternoon: Work, Evening: Stretching'
        }
    }
}


# Fitness challenges dictionary
fitness_challenges = {
    "30-Day Cardio Challenge": {
        "Description": "Improve endurance with progressive cardio workouts.",
        "Intensity": "Moderate",
        "Duration": "30 Days",
        "Exercises": [
            "Day 1-7: 20 mins jogging",
            "Day 8-14: 25 mins running",
            "Day 15-21: 30 mins HIIT",
            "Day 22-30: 35 mins cycling"
        ]
    },
    "Strength Training Challenge": {
        "Description": "Build strength with targeted exercises.",
        "Intensity": "High",
        "Duration": "60 Days",
        "Exercises": [
            "Day 1-15: Push-ups, Squats, Lunges (3 sets each)",
            "Day 16-30: Add Deadlifts, Overhead Press (3 sets each)",
            "Day 31-45: Increase reps by 20%",
            "Day 46-60: Add weights to all exercises"
        ]
    },
    "Core Strength Challenge": {
        "Description": "Focus on your core with a progressive plan.",
        "Intensity": "Moderate",
        "Duration": "30 Days",
        "Exercises": [
            "Day 1-10: Planks (30 seconds), Sit-ups (15 reps)",
            "Day 11-20: Planks (1 min), Sit-ups (20 reps), Russian Twists (15 reps)",
            "Day 21-30: Planks (1.5 min), Sit-ups (25 reps), Bicycle Crunches (20 reps)"
        ]
    },
    "Flexibility Challenge": {
        "Description": "Enhance flexibility with daily stretching.",
        "Intensity": "Low",
        "Duration": "30 Days",
        "Exercises": [
            "Day 1-10: Hamstring Stretch, Cat-Cow Pose (5 mins each)",
            "Day 11-20: Add Butterfly Stretch, Downward Dog (5 mins each)",
            "Day 21-30: Full-body Stretch Routine (15 mins)"
        ]
    },
    "Endurance Boost Challenge": {
        "Description": "Aimed at increasing stamina and overall endurance.",
        "Intensity": "Moderate to High",
        "Duration": "45 Days",
        "Exercises": [
            "Day 1-10: 3 km walk/jog",
            "Day 11-20: 5 km jog/run",
            "Day 21-30: 7 km steady pace run",
            "Day 31-45: Interval training with sprints (30 secs sprint, 1 min jog, repeat 10 times)"
        ]
    },
    "Yoga Mastery Challenge": {
        "Description": "Focus on flexibility, balance, and mindfulness.",
        "Intensity": "Low to Moderate",
        "Duration": "30 Days",
        "Exercises": [
            "Day 1-10: Sun Salutations (5 rounds), Child’s Pose, Downward Dog",
            "Day 11-20: Add Warrior Poses, Tree Pose, Triangle Pose",
            "Day 21-30: Include Headstands (assisted), Crow Pose, and Advanced Stretches"
        ]
    },
    "Fat Burn Challenge": {
        "Description": "High-intensity routines to burn fat efficiently.",
        "Intensity": "High",
        "Duration": "30 Days",
        "Exercises": [
            "Day 1-10: 15 mins HIIT (Jumping Jacks, Burpees, Mountain Climbers)",
            "Day 11-20: 20 mins HIIT + 10 mins steady cardio",
            "Day 21-30: 25 mins HIIT with Tabata-style intervals"
        ]
    },
    "Upper Body Strength Challenge": {
        "Description": "Build strength in the chest, back, and arms.",
        "Intensity": "Moderate to High",
        "Duration": "30 Days",
        "Exercises": [
            "Day 1-10: Push-ups, Pull-ups, Bench Dips (3 sets each)",
            "Day 11-20: Add Dumbbell Rows, Overhead Press (3 sets each)",
            "Day 21-30: Increase reps and add weights for all exercises"
        ]
    },
"Lower Body Strength Challenge": {
    "Description": "Focus on building strength in the legs, glutes, and core.",
    "Intensity": "High",
    "Duration": "30 Days",
    "Exercises": [
        "Day 1-10: 3x10 Squats, 3x10 Lunges, 3x15 Glute Bridges",
        "Day 11-20: 3x12 Squats, 3x12 Lunges, 3x20 Glute Bridges, 3x10 Bulgarian Split Squats",
        "Day 21-30: 3x15 Squats, 3x15 Lunges, 3x25 Glute Bridges, 3x12 Bulgarian Split Squats, 3x10 Deadlifts"
    ]
},

    "Running Speed Challenge": {
        "Description": "Improve running speed and sprinting ability.",
        "Intensity": "High",
        "Duration": "30 Days",
        "Exercises": [
            "Day 1-10: 5x100m sprints with 2 mins rest",
            "Day 11-20: 8x100m sprints with 1.5 mins rest",
            "Day 21-30: 10x100m sprints with 1 min rest"
        ]
    }
}


# App title
st.title("FitFlex")


# Sidebar for user information
st.sidebar.header("Your Information")
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25)
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
gender = st.sidebar.radio("Gender", options=["Male", "Female"])


# Dropdown for selecting a goal
st.header("Select Your Goal")
goal = st.selectbox("Fitness Goal", options=list(fitness_plans.keys()))


# To Generate The Plan Button
if st.button("Let's Go!"):
    # Fetch the selected plan based on gender
    selected_plan = fitness_plans[goal][gender]

    
 # Motivational messages for each goal
    motivations = {
        'Lean': "Consistency is key! Focus on burning calories and maintaining a balanced diet. Every step you take gets you closer to a leaner, healthier you.",
        'Athletic': "Stay strong and agile! Keep pushing your limits with strength and endurance training. You’re building a body that’s ready for anything.",
        'Bulky': "Lift heavy, eat well, and rest smart! Your journey to bulk up is a test of strength and determination. Keep smashing those weights!",
        'Ripped': "You’re carving out a masterpiece! High-intensity workouts and discipline will lead you to that chiseled look. Stay focused!",
        'Jacked': "You’re on your way to beast mode! Combine power and endurance to achieve a physique that stands out. Keep grinding!",
        'Mixed Routine': "Balance is beautiful! By blending strength and cardio, you’re creating a versatile and resilient body. Keep up the great work!",
        'Cut': "Lean and mean! Stick to your calorie deficit and training plan. Your hard work will soon reveal those sharp, defined muscles."
    }

    
    # Display the plan in sections
    st.subheader("Your Fitness Roadmap")
    st.subheader("Exercise Strategy")
    st.write(f"**Exercise Type:** {selected_plan['Exercise Type']}")
    st.write(f"**Exercise Plan:** {selected_plan['Exercise Plan']}")
    st.write(f"**Cardio Minutes:** {selected_plan['Cardio Minutes']}")
    st.write(f"**Strength Training:** {selected_plan['Strength Training']}")

    st.subheader("Hydration and Sleep")
    st.write(f"**Water Intake (Liters):** {selected_plan['Water Intake (Liters)']}")
    st.write(f"**Sleep Hours:** {selected_plan['Sleep Hours']}")

    st.subheader("Diet Plan")
    st.write(f"**Diet Plan:** {selected_plan['Diet Plan']}")
    st.write(f"**Protein (g):** {selected_plan['Protein (g)']}")
    st.write(f"**Carbs (g):** {selected_plan['Carbs (g)']}")
    st.write(f"**Fats (g):** {selected_plan['Fats (g)']}")

    st.subheader("Recommended Fitness Routine")
    st.write(f"**Routine:** {selected_plan['Recommend Fitness Routine']}")

    
    # Display the motivation message for the selected goal
    st.subheader("Before You Begin👇")
    st.write(motivations[goal])


# Fitness Challenges Section
st.header("Challenges to Push Yourself Further💪")
selected_challenge = st.selectbox("Select a Fitness Challenge", options=["None"] + list(fitness_challenges.keys()))


# Display selected challenge details only if a valid challenge is selected
if selected_challenge != "None":
    challenge = fitness_challenges[selected_challenge]
    st.subheader(f"{selected_challenge}")
    st.write(f"**Description:** {challenge['Description']}")
    st.write(f"**Intensity:** {challenge['Intensity']}")
    st.write(f"**Duration:** {challenge['Duration']}")
    st.subheader("Challenge Plan")
    for exercise in challenge["Exercises"]:
        st.write(f"- {exercise}")


# Feedback
feedback = st.radio("Was this App helpful?", ("Yes", "No"), key="feedback_radio")
if feedback == "No":
    feedback_input = st.text_input("Please tell us how we can improve.", key="feedback_input")
    if feedback_input:
        st.write("Thank you for your feedback!")


# To Launch the App (python -m streamlit run Fitness_App_F.py)
