# llm-pelatone

Un LLM trainato con dati italiani al 100% (perché sono un patriota vero italiano vero) e RL in siciliano per atteggiamento e conversazione (soon).

Questo progetto è pensato come un gioco per imparare a costruire un modello di linguaggio basato su dati italiani, con l'obiettivo finale di insegnargli anche il Siciliano stretto stretto e quindi proseguire con un addestramento RL.

## Dataset

Il dataset di base proviene da:

- https://www.corpusitaliano.it/

Questa fonte raccoglie testi in italiano che verranno utilizzati per costruire il corpus di addestramento. Il progetto mira a mantenere un dataset italiano puro e rappresentativo.

## Cosa fa questo progetto

- `lib/dataset.py`: legge i file di testo dalla cartella `data/`, pulisce il testo e costruisce dataset a blocchi di token.
- `lib/models.py`: definisce un piccolo modello Transformer autoregressivo.
- `train.py`: contiene il flusso di training, con spiegazioni passo passo e debug per mostrare come il modello predice il token successivo.
- `app.py`: flusso di sola generazione, usa un checkpoint salvato (`best_model.pt`) per produrre testo a partire da un prompt.

## Come usarlo

### Installare dipendenze e dataset

Assicurati di avere installati Python e gli strumenti di sistema `curl` e `gzip`. Dalla cartella principale del progetto, crea e attiva un virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Installa quindi PyTorch manualmente scegliendo una sola variante:

#### CPU-only

```bash
pip install torch torchvision
```

#### CUDA 12.8

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Infine installa le restanti dipendenze:

```bash
pip install -r requirements.txt
```

Le due varianti di PyTorch sono alternative e non vanno usate insieme. Se CUDA 12.8 non è compatibile con il sistema, usa la variante CPU.

Scarica quindi il dataset ed estrai il corpus principale nella cartella `data/`:

```bash
bash scripts/init.sh
```

Lo script scarica gli archivi PAISA e decomprime `paisa.raw.utf8` direttamente in `data/`. Può essere eseguito dalla cartella principale del progetto o da un'altra directory.

### Preparare manualmente i dati (opzionale)

Se non usi lo script di inizializzazione, metti i file di testo in `data/` con estensione `.utf8` e assicurati che il contenuto sia italiano.

### Allenare il modello

Dopodiché avvia il training:

```bash
python train.py
```

Questo script:

- carica il dataset italiano
- usa il tokenizer GPT-2 con `Tokenizer.from_pretrained("gpt2")`
- crea blocchi di token per l'addestramento
- calcola la loss sulla predizione del token successivo
- salva i checkpoint in `runs/`, compreso `runs/best_model.safetensors` e `runs/final_model.safetensors`

### Generare testo

```bash
python app.py
```

Questo script carica il modello salvato, carica il tokenizer GPT-2 e genera testo a partire da un prompt.

## Piano futuro

- continuare l'addestramento RL (Reinforcement Learning) per migliorare la coerenza e lo stile
- aggiungere il Siciliano e insegnare al modello a parlare Siciliano stretto
- trasformare il progetto in un vero gioco didattico per imparare LLM e linguistica italiana

## Nota

Per ora il modello è un piccolo esperimento locale. Il focus è comprendere la pipeline di training, i dati e la generazione testuale con un LLM in italiano.
E ovviamente questo è un progetto ironico.