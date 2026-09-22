from pathlib import Path
import argparse, random, cv2, joblib, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from feature_extraction import extract_features

EXT={'.jpg','.jpeg','.png','.bmp','.webp'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='PlantVillage color folder')
    ap.add_argument('--max-per-class', type=int, default=300)
    args=ap.parse_args()
    root=Path(args.dataset)
    if not root.exists(): raise SystemExit(f'Dataset folder not found: {root}')
    X=[]; y=[]; rng=random.Random(42)
    classes=[p for p in root.iterdir() if p.is_dir()]
    print(f'Found {len(classes)} class folders.')
    for i,folder in enumerate(sorted(classes),1):
        files=[p for p in folder.iterdir() if p.suffix.lower() in EXT]
        rng.shuffle(files); files=files[:args.max_per_class]
        used=0
        for f in files:
            img=cv2.imread(str(f))
            if img is None: continue
            X.append(extract_features(img)); y.append(folder.name); used+=1
        print(f'[{i}/{len(classes)}] {folder.name}: {used}')
    X=np.asarray(X,dtype=np.float32); y=np.asarray(y)
    if len(set(y))<2: raise SystemExit('Need at least 2 classes.')
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
    models={
      'KNN': Pipeline([('scale',StandardScaler()),('model',KNeighborsClassifier(n_neighbors=3,weights='distance',p=2))]),
      'Logistic Regression': Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=2500,C=2.0,solver='lbfgs'))])
    }
    best_name=None; best=None; best_acc=-1
    for name,m in models.items():
        print(f'Training {name}...'); m.fit(Xtr,ytr); acc=accuracy_score(yte,m.predict(Xte)); print(f'{name} accuracy: {acc:.4f}')
        if acc>best_acc: best_name,best,best_acc=name,m,acc
    Path('models').mkdir(exist_ok=True)
    joblib.dump({'model':best,'model_name':best_name,'accuracy':best_acc,'feature_version':'v2'},'models/best_model.joblib')
    print(f'BEST: {best_name} | accuracy={best_acc:.4f}')
    print('Saved: models/best_model.joblib')

if __name__=='__main__': main()