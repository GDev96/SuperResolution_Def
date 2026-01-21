# Documentazione Tecnica Dettagliata: SwinIR (Swin Transformer for Image Restoration)

Questo repository implementa un framework avanzato per la Super-Risoluzione (SR) di immagini astronomiche basato sull'architettura SwinIR. Il sistema è ottimizzato per l'addestramento distribuito (DDP) e utilizza meccanismi di self-attention su finestre mobili per ricostruire dettagli complessi da osservazioni terrestri, puntando alla qualità del telescopio spaziale Hubble.

## 1. Architettura del Modello (models/)

Il sistema sfrutta la potenza dei Transformer per il restauro delle immagini attraverso moduli specializzati:

* **Generatore SwinIR (architecture.py)**: Implementa l'architettura SwinIR che utilizza il *Swin Transformer Block*. Questo modulo applica la self-attention all'interno di finestre locali e utilizza lo *Shifted Window* per catturare interazioni tra finestre diverse, permettendo di ricostruire strutture nebulari e stellari con estrema precisione.
* **Upsample PixelShuffle (architecture.py)**: Utilizza un modulo di upsampling basato su PixelShuffle che "impara" a distribuire i pixel mancanti, evitando l'effetto sfocato delle interpolazioni classiche.
* **Sistema Discriminatore UNet (discriminator.py)**: Implementa un `UNetDiscriminatorSN` con *Spectral Normalization*. Questa struttura a U analizza l'immagine sia a livello globale che locale, garantendo che le texture generate siano fotometricamente realistiche e libere da artefatti durante il training GAN.

## 2. Gestione Dati e Pipeline (dataset/)

La pipeline è ottimizzata per gestire dati scientifici ad alta fedeltà:

* **astronomical_dataset.py**: Implementa un caricatore che legge immagini TIFF a 16-bit, normalizzando i valori (0-65535) in virgola mobile (0.0-1.0). Questo preserva l'alta gamma dinamica essenziale per non saturare i nuclei stellari.
* **Data Augmentation**: Il dataset applica trasformazioni casuali come flip orizzontali/verticali e rotazioni di 90° (rot90) per aumentare la variabilità dei dati e rendere il modello robusto rispetto all'orientamento del campo inquadrato.

## 3. Logica di Addestramento (train.py)

L'addestramento è configurato per la massima stabilità e scalabilità su più GPU:

* **Distributed Data Parallel (DDP)**: Il sistema utilizza `torch.distributed` per sincronizzare il training su più schede video, ottimizzando i tempi di calcolo.
* **EMA - Exponential Moving Average**: Implementa una classe `ModelEMA` che mantiene una media mobile dei pesi del generatore, producendo un modello finale più stabile e meno sensibile al rumore del training.
* **Funzioni di Perdita (utils/gan_losses.py)**:
    * **CombinedGANLoss**: Integra Pixel Loss (L1), Perceptual Loss (basata su VGG) e Adversarial Loss (RaGAN) con pesi bilanciati per ottenere nitidezza e fedeltà cromatica.
    * **Mixed Precision**: Utilizza `torch.cuda.amp` (GradScaler) per accelerare il training e ridurre l'occupazione di memoria VRAM.

## 4. Inferenza e Analisi (infer.py)

Lo script di inferenza è progettato per la valutazione scientifica e visiva:

* **Auto-Detection dei Parametri**: Lo script rileva automaticamente le dimensioni dell'embedding e la profondità dei layer analizzando lo `state_dict` del checkpoint caricato.
* **Output Scientifico (TIFF 16-bit)**: Salva i risultati della super-risoluzione in formato TIFF a 16-bit per consentire successive analisi fotometriche.
* **Confronto Visivo (Comparison PNG)**: Genera immagini "Tris" che affiancano l'input ingrandito (Nearest Neighbor), il risultato SwinIR e l'originale Hubble per una verifica immediata.

## 5. Setup e Automazione (start.py)

Il file `start.py` funge da launcher interattivo per semplificare la gestione del training:
* **Target Selection**: Permette di selezionare uno o più target celesti (es. M1, M33) combinandone i dataset per un addestramento multi-target.
* **GPU Management**: Rileva le GPU disponibili e consente all'utente di specificare quali ID utilizzare per il processo distribuito.
* **Configurazione Ambiente**: Imposta automaticamente le variabili di sistema per il backend NCCL e le porte di comunicazione distribuita.

## Istruzioni per l'uso

1.  **Requisiti**: Python 3.8+, PyTorch con supporto CUDA.
2.  **Preparazione**: I dati devono essere processati (registrazione/normalizzazione) nel branch `misc` con split JSON pronti.
3.  **Training**: Eseguire `python start.py` e seguire le istruzioni a schermo per selezionare target e GPU.
4.  **Inferenza**: Eseguire `python infer.py`, selezionare la cartella del modello in `outputs/` e generare i risultati TIFF/PNG.
