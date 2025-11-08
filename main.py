from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware


# -------------------- APP CONFIG --------------------
app = FastAPI(title="Mango Disease Detection API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or use ["http://localhost:5173"] for stricter rule
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "mango_resnet50_full_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- CLASS NAMES --------------------
CLASS_NAMES = {
    0: "Anthracnose",
    1: "Bacterial Canker",
    2: "Cutting Weevil",
    3: "Die Back",
    4: "Gall Midge",
    5: "Healthy",
    6: "Powdery Mildew",
    7: "Sooty Mould"
}

# -------------------- CURE & PREVENTION DATA --------------------
DISEASE_INFO = {
    "Anthracnose": {
        "cure_map": [
            "Spray copper-based fungicides like Copper Oxychloride or Mancozeb at early infection.",
            "Remove and destroy infected twigs and leaves to reduce fungal spread.",
            "Ensure proper drainage and avoid water stagnation around the trees."
        ],
        "prevention_map": [
            "Apply preventive sprays of Carbendazim or Mancozeb during flowering and fruiting.",
            "Prune tree canopy to improve air circulation and reduce humidity.",
            "Avoid overhead irrigation to minimize leaf wetness."
        ]
    },
    "Bacterial Canker": {
        "cure_map": [
            "Spray a mixture of Streptomycin Sulfate (200 ppm) + Copper Oxychloride (0.3%) every 10 days.",
            "Remove and burn severely affected plant parts.",
            "Apply Bordeaux paste to wounds or pruning cuts."
        ],
        "prevention_map": [
            "Disinfect pruning tools regularly.",
            "Avoid injuries during harvesting and pruning.",
            "Use disease-free planting material."
        ]
    },
    "Cutting Weevil": {
        "cure_map": [
            "Spray Chlorpyrifos (0.05%) on affected branches and trunk.",
            "Cut and destroy infested shoots to prevent larval spread.",
            "Apply neem oil emulsion (2%) to repel adult weevils."
        ],
        "prevention_map": [
            "Maintain orchard hygiene by clearing fallen debris and dried twigs.",
            "Apply light traps during evening hours to attract and kill adult weevils.",
            "Avoid waterlogging around tree trunks."
        ]
    },
    "Die Back": {
        "cure_map": [
            "Prune infected branches at least 15–20 cm below the infected area and apply Bordeaux paste.",
            "Spray Copper Oxychloride (0.3%) or Mancozeb (0.25%) immediately after pruning.",
            "Repeat fungicidal sprays at 15-day intervals during humid conditions."
        ],
        "prevention_map": [
            "Avoid mechanical injuries to branches.",
            "Ensure balanced fertilization with adequate micronutrients.",
            "Promote good air circulation by proper spacing between trees."
        ]
    },
    "Gall Midge": {
        "cure_map": [
            "Spray Dimethoate (0.05%) or Imidacloprid (0.005%) when galls are first observed.",
            "Remove and destroy affected inflorescences and twigs.",
            "Use sticky traps to monitor adult midge activity."
        ],
        "prevention_map": [
            "Maintain weed-free orchard surroundings.",
            "Encourage biological control agents like parasitoids.",
            "Avoid over-fertilization with nitrogen."
        ]
    },
    "Healthy": {
        "cure_map": ["No treatment required — tree is healthy."],
        "prevention_map": [
            "Maintain regular irrigation and nutrient supply.",
            "Monitor for early signs of disease or pest attacks.",
            "Follow integrated pest management (IPM) practices."
        ]
    },
    "Powdery Mildew": {
        "cure_map": [
            "Spray wettable sulfur (0.2%) or Tridemorph (0.1%) at early stages of infection.",
            "Repeat fungicide application every 10–15 days if infection persists.",
            "Remove infected panicles and leaves."
        ],
        "prevention_map": [
            "Avoid dense canopy to reduce humidity buildup.",
            "Apply preventive sulfur sprays before flowering.",
            "Ensure good sunlight penetration within the tree canopy."
        ]
    },
    "Sooty Mould": {
        "cure_map": [
            "Spray neem oil (2%) or potassium soap solution to wash off mould growth.",
            "Control sap-sucking insects like aphids and mealybugs using Imidacloprid (0.005%).",
            "Prune heavily infested shoots and dispose of them properly."
        ],
        "prevention_map": [
            "Regularly monitor for honeydew-secreting insects.",
            "Encourage natural predators like ladybird beetles.",
            "Maintain cleanliness in the orchard to avoid pest buildup."
        ]
    }
}

# -------------------- MODEL LOADING --------------------
def load_model():
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.eval()
    model.to(device)
    print(f"Model loaded successfully: {type(model)}")
    return model

model = load_model()

# -------------------- IMAGE PREPROCESSING --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------- ROUTES --------------------
@app.get("/")
def home():
    return {"message": "✅ Mango Disease Detection API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Transform and predict
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, preds = torch.max(outputs, 1)
            predicted_class = CLASS_NAMES[preds.item()]

        # Get recommendations
        disease_info = DISEASE_INFO.get(predicted_class, {"cure_map": [], "prevention_map": []})

        return JSONResponse({
            "prediction": predicted_class,
            "cure_recommendations": disease_info["cure_map"],
            "prevention_tips": disease_info["prevention_map"]
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
