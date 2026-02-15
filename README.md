# Super-Risoluzione Astronomica: HST vs Osservatori Terrestri

Un framework completo per la Super-Risoluzione di immagini astronomiche che accoppia osservazioni ad alta risoluzione del telescopio spaziale Hubble (HST) con immagini acquisite da osservatori terrestri (Obs). Il progetto supporta due architetture moderne: **SwinIR** (Swin Transformer) e **HAT-Real** (Hybrid Attention Transformer).

## Panoramica

Questo repository automatizza il processo di creazione di dataset e addestramento di modelli IA per la super-risoluzione di immagini astronomiche. L'obiettivo è accoppiare immagini ad alta risoluzione del telescopio spaziale Hubble con immagini a bassa risoluzione da osservatori terrestri, migliorando la qualità delle osservazioni da terra.

## Struttura del Progetto

Il workflow complessivo è strutturato in tre fasi:

1. **Preparazione Dataset** (cartella `misc/`): Fase comune di allineamento e normalizzazione delle immagini
2. **Scelta dell'Architettura**: Proseguire con SwinIR o HAT per l'addestramento
3. **Training e Inferenza**: Modelli specifici per il restauro della qualità

---

# Fase 1: Preparazione Dataset

Questa fase è **comune a entrambe le architetture** e prepara i dati grezzi per l'addestramento.

### 1. Risoluzione Astrometrica e Registrazione (Dataset_step1_datasetwcs.py)
* **Astrometry Solving**: Utilizza l'eseguibile ASTAP per risolvere le coordinate WCS (World Coordinate System) delle immagini grezze (.fits).
* **Riproiezione**: Allinea le immagini Hubble e quelle dell'osservatorio sulla stessa griglia di coordinate celesti utilizzando la libreria reproject.
* **Output**: Salva le immagini risolte e registrate in cartelle dedicate (2_solved_astap e 3_registered_native).

### 2. Controllo Qualità Mosaico (Dataset_step2_mosaicHSTObs.py)
* **Visualizzazione**: Crea un mosaico di Hubble proiettato sulla tela dell'osservatorio per verificare l'allineamento.
* **Overlay**: Genera un'immagine di confronto RGB (R=Hubble, G=Obs) per identificare eventuali errori di registrazione prima dell'estrazione delle patch.

### 3. Estrazione Patch Dinamica (Dataset_step3_extractpatches.py)
* **Multi-processing**: Utilizza ProcessPoolExecutor per estrarre migliaia di coppie di patch dalle immagini originali.
* **Stride Adattivo**: Calcola automaticamente il passo (stride) di campionamento in base al numero di patch desiderate o a dati storici per bilanciare il dataset.
* **Filtro Copertura**: Scarta le patch con segnale insufficiente basandosi sulla costante MIN_COVERAGE.
* **Check Diagnostico**: Salva immagini di debug per ogni patch per verificare la sovrapposizione a livello di pixel.

### 4. Normalizzazione e Conversione TIFF (Dataset_step4_normalization.py)
* **Log-Stretch**: Applica una trasformazione logaritmica (log1p) per comprimere il range dinamico dei dati FITS.
* **Robust Stats**: Calcola i percentili su tutto il dataset per definire il range dinamico (es. BLACK_CLIP_PERCENTILE).
* **16-bit TIFF**: Converte i dati scientifici a 32-bit in immagini TIFF a 16-bit per i modelli di IA.

### 5. Split del Dataset (prepare_data.py)
* **Train/Val/Test**: Suddivide le coppie di immagini in set di addestramento (80%), validazione (10%) e test (10%).
* **JSON Manifest**: Genera file .json con i percorsi dei file per facilitare il caricamento tramite DataLoader.

## Istruzioni per l'uso

### Prerequisiti
* ASTAP installato nel sistema.
* Librerie Python: numpy, astropy, reproject, tqdm, scikit-image, pillow, torch, matplotlib.

### Esecuzione
Eseguire gli script in ordine numerico all'interno della cartella misc:

1. python Dataset_step1_datasetwcs.py
2. python Dataset_step3_extractpatches.py
3. python Dataset_step4_normalization.py
4. python prepare_data.py

### Configurazione
Le impostazioni principali sono definite come costanti all'inizio di ogni script:
* `HR_SIZE`: 512
* `AI_LR_SIZE`: 128
* `TRAIN_RATIO`: 0.8

Dopo aver completato questi step, i dati sono pronti per l'addestramento con una delle due architetture seguenti.

---

# Fase 2a: SwinIR (Swin Transformer for Image Restoration)

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
2.  **Preparazione**: I dati devono essere processati (registrazione/normalizzazione) nella Fase 1 con split JSON pronti.
3.  **Training**: Eseguire `python start_swin.py` e seguire le istruzioni a schermo per selezionare target e GPU.
4.  **Inferenza**: Eseguire `python infer_swin.py`, selezionare la cartella del modello in `outputs/` e generare i risultati TIFF/PNG.

---

# Fase 2b: HAT-Real (Hybrid Attention Transformer)

Questo repository implementa un framework avanzato per la Super-Risoluzione (SR) di immagini astronomiche, ottimizzato per accoppiare dati di osservatori terrestri con immagini del telescopio spaziale Hubble. Il sistema utilizza un'architettura Hybrid Attention Transformer (HAT) per superare i limiti delle tradizionali reti convoluzionali.

## 1. Architettura del Modello (models/)

Il cuore del sistema è progettato per gestire la complessità dei dati astronomici attraverso tre moduli principali:

* **Generatore HAT (hat_arch.py)**: Implementa il trasformatore ibrido che combina l'efficacia delle convoluzioni nel catturare dettagli locali con la potenza della self-attention per le relazioni globali. Questa struttura permette di ricostruire strutture stellari e nebulose con alta precisione.
* **Sistema Discriminatore (discriminator.py)**: Include diverse architetture per il training avversariale. Utilizza modelli basati su UNet o VGG (srvgg_arch.py) per analizzare l'immagine sia a livello di texture che di struttura complessiva, spingendo il generatore a produrre risultati indistinguibili dalle immagini originali di Hubble.
* **Wrapper Ibrido (hybridmodels.py)**: Una classe di astrazione che facilita la gestione simultanea di generatore e discriminatore, ottimizzando il passaggio dei gradienti e il caricamento dei pesi durante le fasi di addestramento complesso.

## 2. Gestione Dati e Pipeline (dataset/)

La pipeline di caricamento è specificamente progettata per i formati scientifici:

* **astronomical_dataset.py**: Implementa un `DataLoader` personalizzato che legge immagini TIFF a 16-bit. A differenza delle immagini standard a 8-bit, questo preserva l'alta gamma dinamica necessaria per l'astronomia.
* **Data Augmentation**: Il dataset applica trasformazioni casuali (rotazioni di 90°, 180°, 270° e flip orizzontali/verticali) per garantire che il modello sia invariante all'orientamento degli oggetti celesti.

## 3. Logica di Addestramento (train_hat.py)

L'addestramento segue un approccio multi-fase per garantire stabilità e qualità:

* **Funzioni di Perdita (utils/gan_losses.py, losses_train.py)**:
    * **Pixel Loss (L1)**: Assicura che i valori cromatici e di luminosità siano accurati.
    * **Perceptual Loss**: Utilizza una rete VGG pre-addestrata per confrontare le feature estratte, garantendo che le strutture morfologiche siano preservate.
    * **Adversarial Loss**: Fornisce il "tocco finale" per la nitidezza, eliminando l'effetto sfocato tipico delle sole perdite MSE/L1.
* **Metriche di Monitoraggio (utils/metrics.py)**: Vengono calcolati costantemente PSNR (Peak Signal-to-Noise Ratio) e SSIM (Structural Similarity Index) per valutare oggettivamente il miglioramento della qualità.

## 4. Inferenza e Deployment (infer.py)

Lo script di inferenza è ottimizzato per l'uso pratico su dati reali:

* **Processing Batch**: Permette di elaborare intere cartelle di immagini provenienti da sessioni osservative terrestri.
* **Tiling**: Se necessario, gestisce immagini di grandi dimensioni suddividendole in patch per evitare saturazione della memoria GPU, ricomponendole poi senza cuciture visibili.

## 5. Setup e Automazione (start.py)

Il file `start.py` automatizza la preparazione dell'ambiente:
* Verifica la compatibilità dei driver CUDA.
* Configura le cartelle di output per log, checkpoint e immagini campionate.
* Installa le dipendenze mancanti tramite `env_setup.py`.

## Istruzioni per l'uso

1.  **Requisiti**: Python 3.8+, PyTorch 2.0+, GPU con almeno 8GB di VRAM.
2.  **Preparazione**: Assicurarsi che i dati siano stati processati dalla Fase 1 (registrazione e normalizzazione).
3.  **Training**: Eseguire `python train_hat.py` per avviare il ciclo di addestramento.
4.  **Inferenza**: Usare `python infer_hat.py --input <cartella_input> --model_path <checkpoint.pth>` per generare i risultati.

---

## Confronto tra le Architetture

| Aspetto | SwinIR | HAT |
|---------|--------|-----|
| **Approccio** | Swin Transformer puro | Ibrido (Convoluzioni + Transformer) |
| **Self-Attention** | A livello di finestre mobili | Globale e locale |
| **Velocità Training** | Più veloce | Moderata |
| **Precisione** | Eccellente | Superiore |
| **Memoria GPU** | Moderata | Superiore |
| **Ideale per** | Addestramento multi-GPU | Massima qualità scientifica |

---

## Prerequisiti Comuni

- Python 3.8+
- PyTorch con supporto CUDA
- ASTAP (per la registrazione astrometrica)
- Librerie: numpy, astropy, reproject, tqdm, scikit-image, pillow, matplotlib

## Quick Start

```bash
# 1. Preparare il dataset
cd misc/
python Dataset_step1_datasetwcs.py
python Dataset_step3_extractpatches.py
python Dataset_step4_normalization.py
python prepare_data.py

# 2. Scegliere un'architettura e addestrare
# Per SwinIR:
python start_swin.py

# Per HAT:
python train_hat.py
```

