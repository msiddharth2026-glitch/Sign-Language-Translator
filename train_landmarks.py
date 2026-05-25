"""
Production-grade ISL landmark training pipeline.
Uses geometric feature engineering + ensemble (MLP + Random Forest).
"""
import os, sys, pickle, warnings
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import extract_features

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE, "dataset")
WEB_DIR   = os.path.join(BASE, "web")
LANDMARKER = os.path.join(WEB_DIR, "hand_landmarker.task")

# ── Download landmarker ────────────────────────────────────────────────────────
if not os.path.exists(LANDMARKER):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        LANDMARKER
    )

opts = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER),
    num_hands=1, min_hand_detection_confidence=0.3
)
detector = mp_vision.HandLandmarker.create_from_options(opts)


def extract_from_image(path):
    img = cv2.imread(path)
    if img is None: return None

    # Try multiple scales — some training images may be small
    for scale in [1.0, 1.5, 0.75]:
        h, w = img.shape[:2]
        if scale != 1.0:
            img_s = cv2.resize(img, (int(w*scale), int(h*scale)))
        else:
            img_s = img

        # Try original and flipped
        for candidate in [img_s, cv2.flip(img_s, 1)]:
            rgb    = cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res    = detector.detect(mp_img)
            if res.hand_landmarks:
                return extract_features(res.hand_landmarks[0])
    return None


def augment(vec, n=4):
    """Add small noise augmentations."""
    vecs = [vec]
    for _ in range(n):
        noise = np.random.normal(0, 0.008, vec.shape).astype(np.float32)
        vecs.append(vec + noise)
    return vecs


# ── Extract features ───────────────────────────────────────────────────────────
X, y = [], []
categories = sorted(os.listdir(DATA_DIR))
print(f"\nExtracting features from {len(categories)} classes...\n")

for cat in categories:
    if cat == "unknown": continue
    cat_dir = os.path.join(DATA_DIR, cat)
    if not os.path.isdir(cat_dir): continue
    files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    count = 0
    for fname in files:
        vec = extract_from_image(os.path.join(cat_dir, fname))
        if vec is not None:
            for v in augment(vec, n=4):
                X.append(v); y.append(cat)
            count += 1
    print(f"  {cat.upper()}: {count} real samples → {count*5} with augmentation")

X = np.array(X, dtype=np.float32)
print(f"\nTotal: {len(X)} samples, {X.shape[1]} features, {len(set(y))} classes")

# ── Encode labels ──────────────────────────────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"Classes: {list(le.classes_)}")

# ── Scale features ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.15, random_state=42, stratify=y_enc)

# ── MLP model ──────────────────────────────────────────────────────────────────
print("\n--- Training MLP ---")
n_classes = len(le.classes_)
feat_dim  = X.shape[1]

mlp = models.Sequential([
    layers.Input(shape=(feat_dim,)),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(), layers.Dropout(0.35),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(), layers.Dropout(0.25),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(), layers.Dropout(0.2),
    layers.Dense(64,  activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])
mlp.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cb_list = [
    callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'),
    callbacks.ReduceLROnPlateau(patience=7, factor=0.5, min_lr=1e-5),
]
mlp.fit(X_train, y_train, epochs=200, batch_size=64,
        validation_data=(X_test, y_test), callbacks=cb_list, verbose=1)

mlp_loss, mlp_acc = mlp.evaluate(X_test, y_test, verbose=0)
print(f"MLP Test accuracy: {mlp_acc:.4f}")

# ── Random Forest ──────────────────────────────────────────────────────────────
print("\n--- Training Random Forest ---")
rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                             min_samples_split=2, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
rf_acc = rf.score(X_test, y_test)
print(f"RF Test accuracy: {rf_acc:.4f}")

# ── Ensemble prediction ────────────────────────────────────────────────────────
mlp_probs = mlp.predict(X_test, verbose=0)
rf_probs  = rf.predict_proba(X_test)
ensemble  = 0.6 * mlp_probs + 0.4 * rf_probs
ens_preds = np.argmax(ensemble, axis=1)
ens_acc   = np.mean(ens_preds == y_test)
print(f"Ensemble Test accuracy: {ens_acc:.4f}")

# ── Per-class report ───────────────────────────────────────────────────────────
print("\n--- Classification Report ---")
print(classification_report(y_test, ens_preds, target_names=le.classes_))

# ── Confusion: show most confused pairs ───────────────────────────────────────
cm = confusion_matrix(y_test, ens_preds)
confused = []
for i in range(len(le.classes_)):
    for j in range(len(le.classes_)):
        if i != j and cm[i,j] > 0:
            confused.append((cm[i,j], le.classes_[i], le.classes_[j]))
confused.sort(reverse=True)
print("\nTop confused pairs:")
for cnt, a, b in confused[:10]:
    print(f"  {a.upper()} → {b.upper()}: {cnt} times")

# ── Save everything ────────────────────────────────────────────────────────────
mlp.save(os.path.join(BASE, "landmark_model.keras"))
with open(os.path.join(WEB_DIR, "label_encoder.pkl"), "wb") as f: pickle.dump(le, f)
with open(os.path.join(WEB_DIR, "scaler.pkl"),        "wb") as f: pickle.dump(scaler, f)
with open(os.path.join(WEB_DIR, "rf_model.pkl"),      "wb") as f: pickle.dump(rf, f)

print(f"\nAll models saved. Ensemble accuracy: {ens_acc:.4f}")
