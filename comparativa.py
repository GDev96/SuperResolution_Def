import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pandas as pd
import time
import json
import importlib
import math
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import warnings
import torch.nn.functional as F

# Gestione importazione barra di avanzamento
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, desc="", unit=""): return iterable

# ================= SETUP =================
warnings.filterwarnings("ignore")
sns.set_theme(style="white", context="paper")

# --- CONFIGURAZIONE UTENTE ---
VISUAL_SAVE_INTERVAL = 50 
DATASET_BASE_ROOT = Path(r"F:\r\data") 
# -----------------------------

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# ================= CONFIGURAZIONE MODELLI =================

MODELS_CONFIG = {
    "Swin4.0": {
        "code": Path(r"F:\r\swin4"), 
        "ckpt_dir": Path(r"F:\r\swin4\outputs\M1_M33_M42_M8_M82_NGC7635_DDP_SwinIR"),
        "module": "models.architecture",
        "class": "SwinIR",
        "strict_tile": 64,
        "params": {
            "upscale": 4, "in_chans": 1, "img_size": 64, "window_size": 8, "img_range": 1.0,
            "depths": [6, 6, 6, 6, 6, 6], "embed_dim": 180, "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2, "upsampler": "pixelshuffle", "resi_connection": "1conv"
        }
    },
    "HAT_GAN_Light": {
        "code": Path(r"F:\r\purehat"),
        "ckpt_dir": Path(r"F:\r\purehat\outputs\M1_M33_M42_M8_M82_NGC7635"), 
        "module": "models.hat_arch",
        "class": "HAT",
        "strict_tile": None, 
        "params": {} 
    },
    "Hybrid_9.1": {
        "code": Path(r"F:\r\Hybrid"),
        "ckpt_dir": Path(r"F:\r\Hybrid\outputs\M1_M33_M42_M8_M82_NGC7635"),
        "module": "models.hybridmodels",
        "class": "HybridHATRealESRGAN",
        "strict_tile": None,
        "params": {
            "in_chans": 1,      
            "upscale": 4,       
            "num_feat": 48,     
            "num_grow_ch": 24,  
            "embed_dim": 90,    
            "depths": [6, 6, 6, 6],     
            "num_heads": [6, 6, 6, 6]   
        } 
    }
}

# ================= UTILITIES =================

def clean_imports():
    keys = list(sys.modules.keys())
    for k in keys:
        if k == 'models' or k.startswith('models.') or k.startswith('basicsr'):
            del sys.modules[k]

def find_weights_smart(folder):
    if not folder.exists(): return None
    files = list(folder.glob("**/*.pth"))
    if not files: return None
    priority = ['best_gan_model.pth', 'best_hybrid_model.pth', 'latest_checkpoint.pth']
    for name in priority:
        for f in files:
            if f.name == name: return f
    for f in files:
        if 'best' in f.name: return f
    return files[0]

class AstronomicalLoader:
    @staticmethod
    def fix_path(path_str, search_root):
        """
        Corregge i percorsi Linux (/home/user/...) cercandoli nella struttura Windows locale.
        Strategia: Trova la cartella 'pair_XXXXXX' e cerca quella.
        """
        # path_str è tipo: /home/gfrattini/SuperResolution/data/NGC7635/7_dataset_ready_LOG/pair_000786/observatory.tiff
        
        parts = path_str.replace('\\', '/').split('/')
        
        # 1. Cerca l'identificativo "pair_XXXXXX"
        pair_folder = None
        filename = parts[-1]
        
        for part in parts:
            if "pair_" in part:
                pair_folder = part
                break
        
        if pair_folder:
            # Cerca la cartella pair_XXXXXX dentro la root del dataset (es. dentro F:\r\data\NGC7635)
            # rglob cerca ricorsivamente in tutte le sottocartelle
            candidates = list(search_root.rglob(pair_folder))
            if candidates:
                # Abbiamo trovato la cartella pair su Windows!
                # Costruiamo il path completo: .../pair_XXXXXX/observatory.tiff
                return candidates[0] / filename

        # 2. Fallback: prova il path diretto se fosse già corretto
        full_path = search_root / Path(path_str).name
        if full_path.exists(): return full_path
        
        return None 

    @staticmethod
    def load_image(path):
        if path is None: return None
        try:
            img = Image.open(path)
            if img.mode == 'I;16' or path.suffix.lower() in ['.tiff', '.tif']:
                arr = np.array(img, dtype=np.float32) / 65535.0
            else:
                arr = np.array(img.convert('RGB'), dtype=np.float32) / 255.0
            if arr.ndim == 2: arr = np.expand_dims(arr, axis=2) 
            return arr
        except Exception as e:
            return None

def save_visual_comparison(lr, hr, outputs, dataset_name, img_id):
    n_models = len(outputs)
    cols = n_models + 2 
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1: axes = [axes] 

    def show_img(ax, img, title):
        sq_img = np.squeeze(img)
        cmap = 'gray' if sq_img.ndim == 2 else None
        ax.imshow(sq_img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

    show_img(axes[0], lr, f"LR")
    show_img(axes[1], hr, "HR Target")

    for idx, (model_name, sr_img) in enumerate(outputs.items()):
        show_img(axes[idx + 2], sr_img, model_name)

    plt.tight_layout()
    out_dir = Path("confronti_visivi")
    out_dir.mkdir(exist_ok=True)
    
    clean_ds_name = dataset_name.replace("\\", "_").replace("/", "_")
    filename = out_dir / f"{clean_ds_name}_{img_id}.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    return filename

def load_all_datasets(base_root):
    all_pairs = []
    if not base_root.exists():
        print(f"❌ Errore: Cartella dati non trovata: {base_root}")
        return []

    print(f"\n🔍 Scansione dataset in: {base_root}")
    json_files = list(base_root.rglob("test.json"))
    
    if not json_files:
        print(f"❌ Nessun file 'test.json' trovato in {base_root}")
        return []

    for json_path in json_files:
        try:
            relative_path = json_path.relative_to(base_root)
            dataset_name = relative_path.parts[0] 
            search_root = base_root / dataset_name
            
            with open(json_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    item['dataset_name'] = dataset_name
                    item['search_root'] = search_root 
                all_pairs.extend(data)
                print(f"   > Trovato '{dataset_name}': {len(data)} img")
        except Exception as e:
            print(f"   > Errore {json_path}: {e}")
    
    return all_pairs

# ================= CLASSE DI VALUTAZIONE =================

class ModelEvaluator:
    def __init__(self, model, name, weights, device, strict_tile=None, params=None):
        self.model = model.to(device)
        self.name = name
        self.device = device
        self.strict_tile = strict_tile
        
        ckpt = torch.load(weights, map_location=device)
        if 'params_ema' in ckpt: sd = ckpt['params_ema']
        elif 'params' in ckpt: sd = ckpt['params']
        elif 'model_state_dict' in ckpt: sd = ckpt['model_state_dict']
        elif 'net_g' in ckpt: sd = ckpt['net_g']
        else: sd = ckpt
        
        try: self.model.load_state_dict(sd, strict=True)
        except: 
            try: self.model.load_state_dict(sd, strict=False)
            except: pass
        self.model.eval()
        self.in_chans = 3
        if params and 'in_chans' in params: self.in_chans = params['in_chans']
        elif hasattr(self.model, 'conv_first'): self.in_chans = self.model.conv_first.in_channels

    @torch.no_grad()
    def process(self, lr, hr):
        img_chans = lr.shape[2]
        if img_chans != self.in_chans:
            if self.in_chans == 3 and img_chans == 1: lr_in = np.concatenate([lr]*3, axis=2)
            elif self.in_chans == 1 and img_chans == 3: lr_in = np.mean(lr, axis=2, keepdims=True)
            else: lr_in = lr
        else: lr_in = lr

        t_lr = torch.from_numpy(lr_in.transpose(2,0,1)).unsqueeze(0).float().to(self.device)
        if self.device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()

        try:
            if self.strict_tile: out = self.process_tiled(t_lr, crop_sz=self.strict_tile)
            else: out = self.process_padded(t_lr)
        except RuntimeError:
            if self.device.type == 'cuda': torch.cuda.empty_cache()
            return None, 0, 0, 0
            
        if out is None: return None, 0, 0, 0
        if self.device.type == 'cuda': torch.cuda.synchronize()
        dt = time.time()-t0

        sr = out.squeeze(0).cpu().numpy().transpose(1,2,0)
        sr = np.clip(sr, 0, 1)
        if sr.shape[2] == 1 and hr.shape[2] == 3: sr = np.concatenate([sr]*3, axis=2)
        if sr.shape[2] == 3 and hr.shape[2] == 1: sr = np.mean(sr, axis=2, keepdims=True)

        h_min, w_min = min(sr.shape[0], hr.shape[0]), min(sr.shape[1], hr.shape[1])
        sr_c, hr_c = sr[:h_min, :w_min], hr[:h_min, :w_min]
        if hr_c.ndim == 3 and hr_c.shape[2] == 1: hr_c = hr_c.squeeze(2); sr_c = sr_c.squeeze(2)

        p = psnr(hr_c, sr_c, data_range=1.0)
        min_side = min(hr_c.shape[0], hr_c.shape[1])
        win_size = max(3, min(7, min_side - 1 if min_side % 2 == 0 else min_side))
        try: s = ssim(hr_c, sr_c, channel_axis=2 if hr_c.ndim==3 else None, data_range=1.0, win_size=win_size)
        except ValueError: s = 0.0 
        return sr, p, s, dt

    def process_padded(self, t_lr):
        window_size = 64
        _, _, h_old, w_old = t_lr.size()
        h_pad = (window_size - h_old % window_size) % window_size
        w_pad = (window_size - w_old % window_size) % window_size
        if h_pad != 0 or w_pad != 0: t_lr = F.pad(t_lr, (0, w_pad, 0, h_pad), 'reflect')
        out = self.model(t_lr) 
        scale = out.size(2) // t_lr.size(2)
        h_target, w_target = h_old * scale, w_old * scale
        return out[:, :, :h_target, :w_target]

    def process_tiled(self, t_lr, crop_sz):
        b, c, h, w = t_lr.shape
        h_pad = (crop_sz - h % crop_sz) % crop_sz
        w_pad = (crop_sz - w % crop_sz) % crop_sz
        t_lr = F.pad(t_lr, (0, w_pad, 0, h_pad), 'reflect')
        h_new, w_new = t_lr.shape[2], t_lr.shape[3]
        with torch.no_grad(): dummy_out = self.model(t_lr[:, :, :crop_sz, :crop_sz])
        scale = dummy_out.shape[2] // crop_sz
        out_full = torch.zeros((b, dummy_out.shape[1], h_new*scale, w_new*scale), device=self.device)
        for y in range(0, h_new, crop_sz):
            for x in range(0, w_new, crop_sz):
                patch = t_lr[:, :, y:y+crop_sz, x:x+crop_sz]
                patch_out = self.model(patch)
                out_full[:, :, y*scale:(y+crop_sz)*scale, x*scale:(x+crop_sz)*scale] = patch_out
        return out_full[:, :, :h*scale, :w*scale]

# ================= PLOTTING ANALITICO =================

def save_advanced_plots(df, model_params_count):
    if df.empty: return
    # Rimuoviamo eventuali valori nulli o negativi per sicurezza
    df_clean = df[df['PSNR'] > 0].copy()
    
    # Setup del layout
    fig = plt.figure(figsize=(20, 12), dpi=150) # Aumentato DPI per nitidezza
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], wspace=0.3, hspace=0.3)
    
    # 1. DISTRIBUZIONE PSNR (Violin Plot)
    # FIX: inner="box" invece di "stick" per rimuovere l'effetto codice a barre
    ax1 = fig.add_subplot(gs[0, 0])
    sns.violinplot(data=df_clean, x='Model', y='PSNR', hue='Model', 
                   inner="quart", # Mostra i quartili invece di tutte le linee
                   palette="viridis", ax=ax1, legend=False)
    ax1.set_title('Distribuzione PSNR (Densità)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # 2. PERFORMANCE PER DATASET (Box Plot)
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(data=df_clean, x='Dataset', y='PSNR', hue='Model', 
                ax=ax2, palette="Set2", showfliers=False) # showfliers=False nasconde gli outlier estremi per pulizia
    ax2.set_title('PSNR per Target Astronomico', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize='small', framealpha=0.5)
    
    # 3. DIMENSIONI MODELLO (Bar Chart Logaritmico)
    if model_params_count:
        ax3 = fig.add_subplot(gs[0, 2])
        param_df = pd.DataFrame(list(model_params_count.items()), columns=['Model', 'Params'])
        param_df['Params_M'] = param_df['Params'] / 1e6 
        
        # FIX: Scala logaritmica per vedere bene sia i modelli piccoli che grandi
        bars = sns.barplot(data=param_df, x='Model', y='Params_M', palette="Blues_d", ax=ax3)
        ax3.set_yscale('log') # Scala logaritmica
        ax3.set_title('Peso Modello (Scala Log)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Parametri (Milioni)')
        
        # Etichette sopra le barre
        for i, v in enumerate(param_df['Params_M']):
            ax3.text(i, v * 1.1, f"{v:.2f}M", ha='center', fontweight='bold')

    # 4. SSIM MEDIO
    ax4 = fig.add_subplot(gs[1, 0])
    # Aggiungiamo 'errorbar' per vedere la varianza
    sns.barplot(data=df_clean, x='Model', y='SSIM', hue='Model', palette="pastel", ax=ax4, errorbar='sd', legend=False)
    ax4.set_title('SSIM Medio (Structural Similarity)', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 1.0)
    # Nota esplicativa per le GAN
    ax4.text(0.5, -0.15, "Nota: I modelli GAN (Hybrid) hanno SSIM naturalmente più basso\nperché generano dettagli 'finti' non presenti nell'originale.", 
             transform=ax4.transAxes, ha='center', fontsize=8, style='italic', color='gray')

    # 5. TRADE-OFF QUALITÀ / VELOCITÀ (Scatter Plot)
    # FIX: Punti più piccoli e trasparenti per vedere la densità
    ax5 = fig.add_subplot(gs[1, 1:]) # Occupa 2 colonne
    
    # Creiamo un jitter casuale per l'asse X (tempo) per evitare che i punti si sovrappongano verticalmente
    # Il tempo è quasi identico per lo stesso modello, quindi aggiungiamo rumore finto solo per visualizzazione
    df_clean['Time_Jitter'] = df_clean['Time (s)'] * np.random.uniform(0.95, 1.05, size=len(df_clean))
    
    sns.scatterplot(data=df_clean, x='Time_Jitter', y='PSNR', hue='Dataset', style='Model', 
                    s=30, # Punti più piccoli (era 100)
                    alpha=0.6, # Più trasparenza
                    ax=ax5)
    
    ax5.set_xscale('log')
    ax5.set_title('Trade-off: Qualità (PSNR) vs Tempo Inferenza', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Tempo (secondi) - Scala Log')
    ax5.grid(True, which="both", ls="-", alpha=0.2)
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.savefig("5_grafici_dataset_multipli_FIXED.png", dpi=150)
    print(f"\n✅ Grafici Migliorati salvati come '5_grafici_dataset_multipli_FIXED.png'")

# ================= MAIN =================

def main():
    print(f"{Colors.HEADER}--- COMPARATORE MODELLI SR (FIX PATH) ---{Colors.ENDC}")

    # AUTO SELEZIONE HARDWARE (No Menu)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Uso GPU: {Colors.OKBLUE}{torch.cuda.get_device_name(0)}{Colors.ENDC}")
    else:
        device = torch.device('cpu')
        print(f"🔸 Uso CPU")
    print("-" * 30)

    # SELEZIONE MODELLI
    model_keys = list(MODELS_CONFIG.keys())
    print("\nModelli disponibili:")
    for i, name in enumerate(model_keys): print(f" {i+1}. {name}")
    choice = input("\nScelta (INVIO=tutti): ")
    selected_models = {}
    if not choice.strip(): selected_models = MODELS_CONFIG
    else:
        try:
            for idx in [int(x.strip()) - 1 for x in choice.split(',')]:
                if 0 <= idx < len(model_keys): selected_models[model_keys[idx]] = MODELS_CONFIG[model_keys[idx]]
        except: selected_models = MODELS_CONFIG

    # CARICAMENTO MODELLI
    evaluators = {}
    model_params_count = {} 
    print(f"\n{Colors.HEADER}--- CARICAMENTO MODELLI ---{Colors.ENDC}")
    for name, cfg in selected_models.items():
        print(f"🔹 Init {name}...", end=" ")
        if not cfg['code'].exists(): print(f"❌ Path mancante"); continue
        clean_imports(); sys.path.insert(0, str(cfg['code']))
        weights = find_weights_smart(cfg['ckpt_dir'])
        if not weights: weights = find_weights_smart(cfg['code'])
        if not weights: print("❌ Pesi mancanti"); sys.path.pop(0); continue

        try:
            mod = importlib.import_module(cfg['module'])
            cls = getattr(mod, cfg['class'])
            model = cls(**(cfg['params'] if cfg['params'] else {})).to(device)
            model_params_count[name] = sum(p.numel() for p in model.parameters())
            evaluators[name] = ModelEvaluator(model, name, weights, device, cfg.get('strict_tile'), params=cfg['params'])
            print(f"✅ OK ({model_params_count[name]/1e6:.2f}M Params)")
        except Exception as e: print(f"❌ Error: {e}")
        finally: 
            if sys.path[0] == str(cfg['code']): sys.path.pop(0)

    # CARICAMENTO DATASET
    all_pairs = load_all_datasets(DATASET_BASE_ROOT)
    if not all_pairs or not evaluators: return

    detailed_results = []
    print(f"\n{Colors.HEADER}--- AVVIO ELABORAZIONE ({len(all_pairs)} immagini) ---{Colors.ENDC}")
    
    first_error = True

    with tqdm(total=len(all_pairs), desc="Avanzamento", unit="img") as pbar:
        for i, p in enumerate(all_pairs):
            search_root = p.get('search_root', DATASET_BASE_ROOT)
            
            # --- FIX CRITICO: Usa la logica migliorata per trovare le immagini ---
            lr_path = AstronomicalLoader.fix_path(p.get('ground_path',''), search_root)
            hr_path = AstronomicalLoader.fix_path(p.get('hubble_path',''), search_root)
            
            lr = AstronomicalLoader.load_image(lr_path)
            hr = AstronomicalLoader.load_image(hr_path)
            
            if lr is None or hr is None:
                if first_error:
                    tqdm.write(f"{Colors.WARNING}⚠️ Immagine non trovata! Es: {p.get('ground_path','')} in {search_root}{Colors.ENDC}")
                    first_error = False
                pbar.update(1); continue
            
            current_img_outputs = {}
            for ename, ev in evaluators.items():
                sr, p_val, s_val, t_val = ev.process(lr, hr)
                if p_val > 0: 
                    # Usa il nome della cartella 'pair_XXXX' come ID univoco
                    # perché 'observatory.tiff' è uguale per tutti
                    pair_id = lr_path.parent.name if lr_path else f"img_{i}"
                    
                    detailed_results.append({
                        'Model': ename, 'Dataset': p['dataset_name'], 'Image_ID': pair_id,
                        'PSNR': p_val, 'SSIM': s_val, 'Time (s)': t_val
                    })
                    if i % VISUAL_SAVE_INTERVAL == 0: current_img_outputs[ename] = sr
            
            if current_img_outputs:
                save_visual_comparison(lr, hr, current_img_outputs, p['dataset_name'], f"{i}")
            pbar.update(1)

    print(f"\n{Colors.HEADER}--- ANALISI FINALE ---{Colors.ENDC}")
    if detailed_results:
        df = pd.DataFrame(detailed_results)
        print("\n📊 Medie Generali:"); print(df.groupby('Model')[['PSNR', 'SSIM', 'Time (s)']].mean())
        print("\n🎯 Medie per Dataset:"); print(df.groupby(['Dataset', 'Model'])[['PSNR', 'SSIM']].mean())
        df.to_csv("risultati_completi_multidataset.csv", index=False)
        save_advanced_plots(df, model_params_count)
        print(f"\n{Colors.OKGREEN}✅ Immagini salvate in: {Path('confronti_visivi').absolute()}{Colors.ENDC}")
    else:
        print("Nessun risultato valido. Controlla i percorsi.")

if __name__ == "__main__":
    main()
