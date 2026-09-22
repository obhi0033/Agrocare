from pathlib import Path
import argparse, random, cv2, joblib, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from feature_extraction import extract_features

EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='PlantVillage color folder')
    ap.add_argument('--max-per-class', type=int, default=300)
    args = ap.parse_args()

    root = Path(args.dataset)
    if not root.exists():
        raise SystemExit(f'Dataset folder not found: {root}')

if __name__ == '__main__':
    main()