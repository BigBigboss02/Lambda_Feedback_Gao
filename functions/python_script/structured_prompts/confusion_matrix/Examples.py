import json

# Define the examples
examples = {
    "Physics_Energy": [
        "Kinetic energy",
        "Potential energy",
        "Thermal energy",
        "Chemical energy",
        "Nuclear energy",
        "Elastic potential energy",
        "Gravitational potential energy",
        "Sound energy",
        "Light energy",
        "Mechanical energy"
    ],
    "Biology_Levels_Organisation": [
        "Cells",
        "Tissues",
        "Organs",
        "Organ systems",
        "Organisms",
        "Populations",
        "Communities",
        "Ecosystems",
        "Biomes",
        "Biosphere"
    ],
    "Chemistry_Properties_Metals": [
        "Conductivity",
        "Malleability",
        "Ductility",
        "Shiny appearance",
        "High density",
        "High melting point",
        "High boiling point",
        "Sonority",
        "Opacity",
        "Thermal conductivity"
    ],
    "Mathematics_Types_Numbers": [
        "Natural numbers",
        "Whole numbers",
        "Integers",
        "Rational numbers",
        "Irrational numbers",
        "Real numbers",
        "Complex numbers",
        "Prime numbers",
        "Composite numbers",
        "Transcendental numbers"
    ],
    "Physics_Fundamental_Forces": [
        "Gravitational force",
        "Electromagnetic force",
        "Strong nuclear force",
        "Weak nuclear force",
        "Frictional force",
        "Tension force",
        "Normal force",
        "Air resistance",
        "Applied force",
        "Spring force"
    ],
    "Biology_Components_Cell": [
        "Nucleus",
        "Mitochondria",
        "Ribosomes",
        "Endoplasmic reticulum",
        "Golgi apparatus",
        "Lysosomes",
        "Chloroplasts",
        "Cell membrane",
        "Cell wall",
        "Cytoplasm"
    ],
    "Chemistry_Chemical_Bonds": [
        "Ionic bond",
        "Covalent bond",
        "Metallic bond",
        "Hydrogen bond",
        "Van der Waals forces",
        "Polar covalent bond",
        "Nonpolar covalent bond",
        "Coordinate bond",
        "Sigma bond",
        "Pi bond"
    ],
    "Mathematics_Shapes_Geometry": [
        "Triangle",
        "Square",
        "Rectangle",
        "Circle",
        "Polygon",
        "Pentagon",
        "Hexagon",
        "Octagon",
        "Ellipse",
        "Parallelogram"
    ],
    "Physics_Types_Waves": [
        "Longitudinal waves",
        "Transverse waves",
        "Electromagnetic waves",
        "Mechanical waves",
        "Surface waves",
        "Sound waves",
        "Radio waves",
        "Light waves",
        "Infrared waves",
        "Microwaves"
    ],
    "Biology_Human_Digestive_System": [
        "Mouth",
        "Esophagus",
        "Stomach",
        "Small intestine",
        "Large intestine",
        "Rectum",
        "Anus",
        "Liver",
        "Gallbladder",
        "Pancreas"
    ]
}

# Save as JSON file
file_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\functions\python_script\structured_prompts\confusion_matrix\A_Level_STEM_Answers.json"
with open(file_path, "w") as json_file:
    json.dump({"example_text": examples}, json_file, indent=4)

file_path
# Define a function to generate examples for each subject

def generate_examples(subject, answers):
    examples = [
        {
            "input": f"List 3 types of {subject.replace('_', ' ').lower()}.",
            "output": f"1. {answers[0]}, 2. {answers[1]}, 3. {answers[2]}.",
            "correct": True
        },
        {
            "input": f"List 3 types of {subject.replace('_', ' ').lower()}.",
            "output": f"1. {answers[3]}, 2. {answers[4]}, 3. {answers[5]}.",
            "correct": True
        },
        {
            "input": f"List 3 types of {subject.replace('_', ' ').lower()}.",
            "output": f"1. {answers[0]}, 2. {answers[1]}.",
            "correct": False
        },
        {
            "input": f"List 3 types of {subject.replace('_', ' ').lower()}.",
            "output": "1. 12345, 2. 67890, 3. 24680.",
            "correct": False
        }
    ]
    return examples

# Generate examples for all subjects
all_examples = {}
for subject, answers in examples.items():
    all_examples[subject] = generate_examples(subject, answers)

# Save as JSON file
file_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\functions\python_script\structured_prompts\confusion_matrix\A_Level_STEM_Examples.json"
with open(file_path, "w") as json_file:
    json.dump({"examples_with_correctness": all_examples}, json_file, indent=4)

file_path
