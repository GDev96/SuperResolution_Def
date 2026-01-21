# Pipeline di Training e Inferenza: HAT-Real (Hybrid Attention Transformer)

Questo branch contiene l'implementazione del modello Hybrid Attention Transformer (HAT) applicato alla super-risoluzione di immagini astronomiche Real-World. Il codice gestisce l'intero ciclo di vita del modello, dal caricamento dei dati all'addestramento con perdite GAN, fino all'inferenza su nuove immagini.

## Architettura del Progetto

Il branch è organizzato nelle seguenti componenti principali:

### 1. Core del Modello (models/)
* **hat_arch.py**: Implementazione del trasformatore ibrido che combina self-attention ed estrazione di feature convoluzionali per catturare dettagli locali e globali.
* **discriminator.py**: Architettura del discriminatore basata su UNet o VGG (srvgg_arch.py) per l'addestramento di tipo Generative Adversarial Network (GAN).
* **hybridmodels.py**: Classi wrapper che integrano il generatore e il discriminatore per facilitare le operazioni di training.

### 2. Gestione Dati (dataset/)
* **astronomical_dataset.py**: Caricatore personalizzato per coppie di immagini Hubble (HR) e Osservatorio (LR). Gestisce la lettura dei file TIFF a 16-bit prodotti dalla pipeline di preprocessing e applica data augmentation (rotazioni, flip).

### 3. Logica di Addestramento (train_hat.py)
* **Training Loop**: Gestisce l'ottimizzazione simultanea di generatore e discriminatore.
* **Loss Functions (utils/losses_train.py e gan_losses.py)**: Integra diverse funzioni di perdita, tra cui L1 loss, Perceptual loss (basata su VGG) e Adversarial loss per migliorare il realismo fotometrico.
* **Validazione**: Calcola metriche come PSNR e SSIM (utils/metrics.py) durante l'addestramento per monitorare la convergenza.

### 4. Inferenza e Deployment (infer.py)
* **Script di Test**: Carica i pesi del modello pre-addestrato ed esegue la super-risoluzione su singole immagini o interi set di test, salvando i risultati per il confronto visivo.

### 5. Utility e Setup (start.py e utils/env_setup.py)
* **Automazione**: start.py funge da entry-point per configurare l'ambiente, verificare le dipendenze e avviare il processo di training o inferenza in base ai parametri passati.

## Istruzioni per l'uso

### Prerequisiti
* Ambiente Python 3.8 o superiore.
* GPU compatibile con CUDA (fortemente raccomandata per il training).
* Installazione dipendenze:
  ```bash
  pip install -r requirements.txt
  ```
### Esecuzione Training
Per avviare l'addestramento del modello HAT:

```Bash
python train_hat.py
```
### Esecuzione Inferenza
Per applicare il modello a nuove immagini:

```Bash
python infer.py --input path/to/images --model_path path/to/checkpoint.pth
```

### Note Tecniche
Dataset: Il modello si aspetta dati normalizzati e suddivisi tramite gli script presenti nel branch di preprocessing (Dataset_step4 e prepare_data).

Configurazione: I parametri relativi a learning rate, batch size e pesi delle loss sono definiti all'interno di train_hat.py e start.py.
