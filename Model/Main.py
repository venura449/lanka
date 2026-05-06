import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────
# 1. THRESHOLD CONSTANTS
# ─────────────────────────────────────────────
COOLANT_NORMAL = (90, 105)
COOLANT_WARN = 108
COOLANT_CRIT = 115

OIL_NORMAL = (20, 60)
OIL_WARN_LOAD = 15
OIL_CRIT = 10

MAP_NORMAL = (30, 50)
MAP_WARN_KPA = 27.1
MAP_CRIT_KPA = 101.3

RPM_NORMAL = (600, 1000)
RPM_REDLINE = 6500
RPM_CRIT = RPM_REDLINE * 1.10


# ─────────────────────────────────────────────
# 2. LABEL FUNCTION
# ─────────────────────────────────────────────
def label_row(coolant, oil, map_kpa, rpm):
    if coolant > COOLANT_CRIT:
        return "CRIT"
    if oil < OIL_CRIT:
        return "CRIT"
    if map_kpa >= MAP_CRIT_KPA:
        return "CRIT"
    if rpm > RPM_CRIT:
        return "CRIT"

    if coolant > COOLANT_WARN:
        return "WARN"

    under_load = rpm > 1000

    if oil < OIL_WARN_LOAD and under_load:
        return "WARN"
    if map_kpa < MAP_WARN_KPA and rpm <= 1000:
        return "WARN"

    return "GOOD"


# ─────────────────────────────────────────────
# 3. SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────
def generate_dataset(n=500):
    rows = []

    targets = {
        "GOOD": int(n * 0.60),
        "WARN": int(n * 0.25),
        "CRIT": n - int(n * 0.60) - int(n * 0.25),
    }

    def make_good():
        coolant = np.random.uniform(90, 105)
        oil = np.random.uniform(20, 60)
        map_kpa = np.random.uniform(30, 50)
        rpm = np.random.uniform(600, 1000)
        return coolant, oil, map_kpa, rpm

    def make_warn():
        scenario = np.random.choice(["hot", "low_oil_load", "vacuum"])

        if scenario == "hot":
            coolant = np.random.uniform(108.01, 114.99)
            oil = np.random.uniform(15, 60)
            map_kpa = np.random.uniform(30, 50)
            rpm = np.random.uniform(600, 1500)

        elif scenario == "low_oil_load":
            coolant = np.random.uniform(90, 108)
            oil = np.random.uniform(10.01, 14.99)
            map_kpa = np.random.uniform(35, 60)
            rpm = np.random.uniform(1001, 3000)

        else:
            coolant = np.random.uniform(90, 108)
            oil = np.random.uniform(15, 60)
            map_kpa = np.random.uniform(20, 27.09)
            rpm = np.random.uniform(600, 1000)

        return coolant, oil, map_kpa, rpm

    def make_crit():
        scenario = np.random.choice(["overheat", "no_oil", "boost", "overrev"])

        if scenario == "overheat":
            coolant = np.random.uniform(115.01, 130)
            oil = np.random.uniform(10.01, 60)
            map_kpa = np.random.uniform(30, 70)
            rpm = np.random.uniform(600, 4000)

        elif scenario == "no_oil":
            coolant = np.random.uniform(90, 130)
            oil = np.random.uniform(0, 9.99)
            map_kpa = np.random.uniform(30, 70)
            rpm = np.random.uniform(600, 5000)

        elif scenario == "boost":
            coolant = np.random.uniform(90, 115)
            oil = np.random.uniform(10, 60)
            map_kpa = np.random.uniform(101.3, 160)
            rpm = np.random.uniform(1500, 6000)

        else:
            coolant = np.random.uniform(90, 115)
            oil = np.random.uniform(20, 60)
            map_kpa = np.random.uniform(40, 80)
            rpm = np.random.uniform(RPM_CRIT + 1, RPM_CRIT + 1000)

        return coolant, oil, map_kpa, rpm

    generators = {
        "GOOD": make_good,
        "WARN": make_warn,
        "CRIT": make_crit,
    }

    for target_label, count in targets.items():
        generated = 0
        attempts = 0

        while generated < count:
            c, o, m, r = generators[target_label]()
            actual = label_row(c, o, m, r)

            if actual == target_label:
                rows.append({
                    "coolant_c": round(c, 2),
                    "oil_psi": round(o, 2),
                    "map_kpa": round(m, 2),
                    "rpm": int(r),
                    "status": actual,
                })
                generated += 1

            attempts += 1
            if attempts > count * 200:
                break

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()

    df["coolant_over_warn"] = (df["coolant_c"] - COOLANT_WARN).clip(lower=0)
    df["coolant_over_crit"] = (df["coolant_c"] - COOLANT_CRIT).clip(lower=0)

    df["oil_deficit"] = (OIL_WARN_LOAD - df["oil_psi"]).clip(lower=0)
    df["oil_danger"] = (OIL_CRIT - df["oil_psi"]).clip(lower=0)

    df["map_vacuum_deficit"] = (MAP_WARN_KPA - df["map_kpa"]).clip(lower=0)
    df["map_boost_excess"] = (df["map_kpa"] - MAP_CRIT_KPA).clip(lower=0)

    df["rpm_over_redline"] = (df["rpm"] - RPM_REDLINE).clip(lower=0)
    df["under_load"] = (df["rpm"] > 1000).astype(int)
    df["oil_load_risk"] = df["oil_deficit"] * df["under_load"]

    return df


# ─────────────────────────────────────────────
# 5. TRAIN AND EVALUATE
# ─────────────────────────────────────────────
def train_and_evaluate(df):
    df_feat = engineer_features(df)

    FEATURES = [
        "coolant_c",
        "oil_psi",
        "map_kpa",
        "rpm",
        "coolant_over_warn",
        "coolant_over_crit",
        "oil_deficit",
        "oil_danger",
        "map_vacuum_deficit",
        "map_boost_excess",
        "rpm_over_redline",
        "under_load",
        "oil_load_risk",
    ]

    le = LabelEncoder()

    X = df_feat[FEATURES]
    y = le.fit_transform(df_feat["status"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("=" * 60)
    print("  ENGINE HEALTH CLASSIFIER — TRAINING REPORT")
    print("=" * 60)

    best_model = None
    best_name = None
    best_acc = 0.0

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        print(f"\n── {name} ──")
        print(f"  CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")

        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        print("  Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_string())

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model
            best_name = name

    print("\n" + "=" * 60)
    print(f"  BEST MODEL : {best_name}  (Test Acc = {best_acc:.4f})")
    print("=" * 60)

    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=FEATURES)
        fi = fi.sort_values(ascending=False)

        print("\n  Top Feature Importances:")
        for feat, imp in fi.head(8).items():
            bar = "█" * int(imp * 50)
            print(f"    {feat:<25} {imp:.4f}  {bar}")

    return best_model, le, FEATURES, best_name, best_acc


# ─────────────────────────────────────────────
# 6. SAVE MODEL FILES
# ─────────────────────────────────────────────
def save_model(model, label_encoder, features):
    joblib.dump(model, "engine_health_model.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    joblib.dump(features, "features.pkl")

    print("\nModel files saved:")
    print(" - engine_health_model.pkl")
    print(" - label_encoder.pkl")
    print(" - features.pkl")


# ─────────────────────────────────────────────
# 7. INFERENCE HELPER
# ─────────────────────────────────────────────
def predict_status(model, le, features, coolant, oil, map_kpa, rpm):
    row = pd.DataFrame([{
        "coolant_c": coolant,
        "oil_psi": oil,
        "map_kpa": map_kpa,
        "rpm": rpm,

        "coolant_over_warn": max(0, coolant - COOLANT_WARN),
        "coolant_over_crit": max(0, coolant - COOLANT_CRIT),

        "oil_deficit": max(0, OIL_WARN_LOAD - oil),
        "oil_danger": max(0, OIL_CRIT - oil),

        "map_vacuum_deficit": max(0, MAP_WARN_KPA - map_kpa),
        "map_boost_excess": max(0, map_kpa - MAP_CRIT_KPA),

        "rpm_over_redline": max(0, rpm - RPM_REDLINE),
        "under_load": int(rpm > 1000),
        "oil_load_risk": max(0, OIL_WARN_LOAD - oil) * int(rpm > 1000),
    }])

    pred = model.predict(row[features])[0]
    proba = model.predict_proba(row[features])[0]

    label = le.inverse_transform([pred])[0]
    conf = {le.classes_[i]: round(float(p), 3) for i, p in enumerate(proba)}

    return label, conf


# ─────────────────────────────────────────────
# 8. LOAD MODEL EXAMPLE
# ─────────────────────────────────────────────
def load_saved_model():
    model = joblib.load("engine_health_model.pkl")
    le = joblib.load("label_encoder.pkl")
    features = joblib.load("features.pkl")
    return model, le, features


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating 500-row dataset ...")

    df = generate_dataset(500)

    print(f"Dataset shape : {df.shape}")
    print(f"Label distribution:\n{df['status'].value_counts().to_string()}\n")

    df.to_csv("engine_health_dataset.csv", index=False)
    print("Dataset saved -> engine_health_dataset.csv\n")

    best_model, le, features, best_name, best_acc = train_and_evaluate(df)

    save_model(best_model, le, features)

    print("\n" + "=" * 60)
    print("  DEMO — ORIGINAL LOG LINES RE-EVALUATED BY MODEL")
    print("=" * 60)

    log_samples = [
        (117.47, 6.26, 99.08, 1486, "CRIT"),
        (112.95, 11.24, 55.01, 1110, "WARN"),
        (114.57, 12.32, 51.53, 1039, "WARN"),
        (102.59, 23.14, 46.59, 782, "GOOD"),
        (93.63, 46.42, 48.29, 814, "GOOD"),
        (104.44, 31.74, 35.90, 838, "GOOD"),
        (113.24, 11.62, 57.43, 1018, "WARN"),
        (102.16, 24.21, 34.37, 938, "GOOD"),
        (91.91, 54.87, 32.08, 879, "GOOD"),
    ]

    print(
        f"{'Coolant':>8} {'Oil':>6} {'MAP':>7} {'RPM':>5}  "
        f"{'Actual':<6}  {'Predicted':<10}  Confidence"
    )
    print("-" * 70)

    for c, o, m, r, actual in log_samples:
        pred, conf = predict_status(best_model, le, features, c, o, m, r)
        match = "✓" if pred == actual else "✗"

        print(
            f"{c:>8.2f} {o:>6.2f} {m:>7.2f} {r:>5}  "
            f"{actual:<6}  {pred:<10} {match}  {conf}"
        )

    print("\nDone.")
    print("Files created:")
    print(" - engine_health_dataset.csv")
    print(" - engine_health_model.pkl")
    print(" - label_encoder.pkl")
    print(" - features.pkl")