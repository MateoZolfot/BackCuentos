import os, torch, torchvision.models as models
from torchvision import transforms
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "models"

WEIGHTS = BASE_DIR / "resnet18_places365.pth.tar"
CATS_TXT = BASE_DIR / "categories_places365.txt"

def _download(url: str, dest: Path):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.system(f"wget -q {url} -O {dest}")
        
ARCH = 'resnet18'
WEIGHTS = f'{ARCH}_places365.pth.tar'
CATS_TXT = 'categories_places365.txt'

# ---------- descarga pesos / txt si faltan ----------
def _download(url, fname):
    if not os.path.exists(fname):
        os.system(f"wget -q {url} -O {fname}")

def load_places_model():
    _download(f'http://places2.csail.mit.edu/models_places365/{WEIGHTS}', WEIGHTS)
    _download('https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt', CATS_TXT)

    model = models.__dict__[ARCH](num_classes=365)
    ckpt = torch.load(WEIGHTS, map_location='cpu')
    state = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state)
    model.eval()
    return model

def load_categories():
    with open(CATS_TXT) as f:
        return [line.strip().split(' ')[0][3:] for line in f]

MODEL = load_places_model()
CLASSES = load_categories()

TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_scene(img: Image.Image):
    x = TRANSFORM(img).unsqueeze(0)
    with torch.inference_mode():
        logits = MODEL(x)
        probs = torch.softmax(logits, 1)[0]
    conf, idx = probs.max(0)
    return CLASSES[idx], float(conf)
