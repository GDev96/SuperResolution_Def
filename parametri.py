import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF_functional
from pathlib import Path


sys.modules['torchvision.transforms.functional_tensor'] = TF_functional


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

try:
    from models.hybridmodels import HybridHATRealESRGAN
    from models.discriminator import UNetDiscriminatorSN
except ImportError as e:
    sys.exit(f" Errore Import: {e}")

MODEL_CONFIG = {
    "img_size": 128,
    "in_chans": 1,
    "embed_dim": 180,
    "depths": (6, 6, 6, 6, 6, 6),
    "num_heads": (6, 6, 6, 6, 6, 6),
    "window_size": 8,
    "upscale": 4,
    "num_rrdb": 23,
    "num_feat": 64,
    "num_grow_ch": 32
}

def print_row(name, params):
    mb_size = (params * 4) / (1024 * 1024)
    print(f" │ {name:<25} │ {params:>12,} │ {mb_size:>9.2f} MB │")

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n🔨 Calcolo parametri in corso...\n")

    try:
    
        net_g = HybridHATRealESRGAN(**MODEL_CONFIG)
        net_d = UNetDiscriminatorSN(num_in_ch=1, num_feat=64)

     
        hat_params = sum(p.numel() for p in net_g.hat.parameters())
        
     
        disc_params = sum(p.numel() for p in net_d.parameters())
        

        total_gen = sum(p.numel() for p in net_g.parameters())
        grand_total = total_gen + disc_params

        # 3. Stampa
        print(" ┌───────────────────────────────────────────────────────┐")
        print(" │ RIEPILOGO PARAMETRI (HAT, DISC, TOT)                  │")
        print(" ├───────────────────────────────────────────────────────┤")
        print(" │ Componente                │    Parametri │ Dimensione │")
        print(" ├───────────────────────────────────────────────────────┤")
        
        print_row("Parte HAT (Generator)", hat_params)
        print_row("Discriminatore", disc_params)
        
        print(" ╞═══════════════════════════════════════════════════════╡")
        print_row("TOTALE SISTEMA", grand_total)
        print(" └───────────────────────────────────────────────────────┘")
       

    except Exception as e:
        print(f" Errore: {e}")

if __name__ == "__main__":
    main()
