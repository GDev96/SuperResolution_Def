import torch
from models.architecture import SwinIR
from models.discriminator import UNetDiscriminatorSN

def count_parameters(model):
    """Conta i parametri addestrabili di un modello Pytorch."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print("Calcolo dei parametri dei modelli...\n")

    # 1. Configurazione del Generatore (SwinIR)
    # Parametri presi dal tuo file train.py per rispecchiare il training attuale
    net_g = SwinIR(
        upscale=4,
        in_chans=1,
        img_size=128,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )

    # 2. Configurazione del Discriminatore
    # Parametri presi dal tuo file train.py
    net_d = UNetDiscriminatorSN(
        num_in_ch=1, 
        num_feat=64
    )

    # 3. Calcolo
    g_params = count_parameters(net_g)
    d_params = count_parameters(net_d)

    # 4. Stampa Risultati
    print("-" * 40)
    print(f"GENERATORE (SwinIR)")
    print(f"Parametri totali: {g_params:,}")
    print(f"In Milioni (M):   {g_params/1e6:.2f} M")
    print("-" * 40)
    
    print(f"DISCRIMINATORE (UNetDiscriminatorSN)")
    print(f"Parametri totali: {d_params:,}")
    print(f"In Milioni (M):   {d_params/1e6:.2f} M")
    print("-" * 40)
    
    print(f"TOTALE SISTEMA:   {(g_params + d_params)/1e6:.2f} M")

if __name__ == "__main__":
    main()
