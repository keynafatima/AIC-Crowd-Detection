from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil, os
from inference import infer_on_image_path, infer_on_image_bytes
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/web", StaticFiles(directory="web"), name="web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

@app.get("/")
async def root():
    return FileResponse("web/index.html")

# ---------------- KERETA ----------------
@app.get("/predict-kereta")
def predict_kereta():
    results = []
    best_carriage = None
    min_count = 999999

    kapasitas = 40  # misal kapasitas 40 orang per gerbong

    for f in os.listdir(DATA_DIR):
        if f.lower().endswith((".jpg",".png",".jpeg")):
            path = os.path.join(DATA_DIR, f)
            r = infer_on_image_path(path)

            base = os.path.splitext(os.path.basename(f))[0]
            out_file = os.path.join("static", f"out_{base}.jpg")
            shutil.copy(r["heatmap"], out_file)
            r["out_file"] = f"static/out_{base}.jpg"

            r["occupancy"] = round((r["count"]/kapasitas)*100,1)

            results.append(r)

            if r["count"] < min_count:
                min_count = r["count"]
                best_carriage = r.get("carriage", f)

    rekom = f"Naik di gerbong {best_carriage}, karena lebih sepi."
    return {
        "results": results,
        "best_carriage": best_carriage,
        "occupancy": min_count,
        "desc": "Analisis jumlah orang dari semua gerbong.",
        "recommendation": rekom
    }

# ---------------- BUS POST ----------------
@app.post("/predict-bus")
async def predict_bus_post(file: UploadFile = File(...)):
    contents = await file.read()
    r = infer_on_image_bytes(contents, name=file.filename)

    out_file = os.path.join("static", f"out_{file.filename}")
    with open(out_file, "wb") as f:
        f.write(r["image_bytes"])

    kapasitas = 40
    percent = round((r["count"]/kapasitas)*100,1)

    return {
        "count": r["count"],
        "occupancy": percent,
        "recommendation": "Bus penuh!" if percent > 70 else "Bus masih cukup kosong",
        "out_file": f"static/out_{file.filename}"
    }

# ---------------- BUS GET ----------------
@app.get("/predict-bus")
def predict_bus_get():
    default_path = os.path.join(DATA_DIR, "krl8.jpeg")
    r = infer_on_image_path(default_path)
    filename = "krl8.jpeg"

    out_file = os.path.join("static", f"out_{filename}")
    shutil.copy(r["heatmap"], out_file)

    kapasitas = 40
    percent = round((r["count"]/kapasitas)*100,1)

    return {
        "count": r["count"],
        "occupancy": percent,
        "recommendation": "Bus penuh!" if percent > 70 else "Bus masih cukup kosong",
        "out_file": f"static/out_{filename}"
    }
