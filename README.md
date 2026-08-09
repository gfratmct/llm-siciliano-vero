# llm-pelatone

Un LLM trainato con dati italiani al 100% (perché sono un patriota vero italiano vero) e RL in siciliano per atteggiamento e conversazione (soon).

Questo progetto è pensato come un gioco per imparare a costruire un modello di linguaggio basato su dati italiani, con l'obiettivo finale di insegnargli anche il Siciliano stretto stretto e quindi proseguire con un addestramento RL.

## Dataset

Il dataset di base proviene da:

- https://www.corpusitaliano.it/

Questa fonte raccoglie testi in italiano che verranno utilizzati per costruire il corpus di addestramento. Il progetto mira a mantenere un dataset italiano puro e rappresentativo.

Come riferimento per ampliare il corpus con più dati grezzi in italiano:

- https://dumps.wikimedia.org/itwiki/ (dump completo di Wikipedia in italiano)

Lo snapshot di Wikipedia italiana è uno dei corpus in lingua italiana più grandi e variegati disponibili gratuitamente. Puoi scaricare il dump `pages-articles` e processarlo con tool come `wikiextractor` per estrarre il testo pulito da aggiungere alla cartella `data/`.

## Cosa fa questo progetto

- `lib/dataset.py`: legge i file di testo dalla cartella `data/`, pulisce il testo e costruisce dataset a blocchi di token. Tutta la fase di lettura, pulizia e tokenizzazione mostra barre di avanzamento (`tqdm`).
- `lib/models.py`: definisce un piccolo modello Transformer autoregressivo, più la funzione `resize_token_embeddings` per espandere il vocabolario quando si aggiungono token speciali.
- `lib/tokenizer.py`: wrapper attorno al tokenizer (BPE byte-level) con i token speciali per la chat (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`, `<pad>`, `<unk>`). Carica automaticamente un tokenizer generato su misura se esiste `lib/tokenizer.json`, altrimenti usa GPT-2 come fallback.
- `scripts/train_tokenizer.py`: addestra un tokenizer BPE sul corpus italiano e lo salva in `lib/tokenizer.json`.
- `train.py`: contiene il flusso di training, con spiegazioni passo passo e debug per mostrare come il modello predice il token successivo.
- `app.py`: flusso di sola generazione, usa un checkpoint salvato per produrre testo a partire da un prompt.

## Come usarlo

### Installare dipendenze e dataset

Assicurati di avere installati Python **3.12**, `curl` e `gzip`. Dalla cartella principale del progetto, crea e attiva un virtualenv con Python 3.12:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
```

Se non usi `uv`, crea un virtualenv con il tuo Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Installa quindi PyTorch **2.8.0** manualmente scegliendo la variante adatta al tuo sistema:

#### macOS (Apple Silicon / MPS)

```bash
pip install torch==2.8.0 torchvision==0.23.0
```

#### Linux + NVIDIA (CUDA 12.8)

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

#### CPU-only (Linux/Windows senza GPU)

```bash
pip install torch==2.8.0 torchvision==0.23.0
```

Infine installa le restanti dipendenze:

```bash
pip install -r requirements.txt
```

Le varianti di PyTorch sono alternative e non vanno usate insieme. Su macOS non esiste CUDA: si usa il backend **MPS** (verifica con `python -c "import torch; print(torch.backends.mps.is_available())"`). L'index `cu128` contiene solo wheel Linux.

Scarica quindi il dataset ed estrai il corpus principale nella cartella `data/`:

```bash
bash scripts/init.sh
```

Lo script scarica gli archivi PAISA e decomprime `paisa.raw.utf8` direttamente in `data/`. Può essere eseguito dalla cartella principale del progetto o da un'altra directory.

### Preparare manualmente i dati (opzionale)

Se non usi lo script di inizializzazione, metti i file di testo in `data/` con estensione `.utf8` e assicurati che il contenuto sia italiano. I file `.parquet` (es. dump di Wikipedia) vanno posizionati in una sottocartella di `data/` (es. `data/20231101.it/`).

### Addestrare il tokenizer (consigliato)

Prima di allenare il modello, genera un tokenizer BPE addestrato sul tuo corpus italiano.Questo produce token più efficienti per l'italiano rispetto al tokenizer GPT-2 (che è ottimizzato per l'inglese) e include già i token speciali per la chat.

```bash
python train_tokenizer.py
```

Opzioni disponibili:

```bash
mkdir -p models/
python train_tokenizer.py --vocab-size 50257 --data-dir data/ --output models/tokenizer.json
```

Il tokenizer viene salvato in `lib/tokenizer.json` e viene caricato automaticamente da `lib/tokenizer.py` in tutti gli script del progetto. Durante l'addestramento del tokenizer vedrai barre di avanzamento per la lettura del corpus e le fasi interne del BPE (pre-processing, tokenize words, count pairs, compute merges).

Se non addestri un tokenizer personalizzato, il progetto usa il fallback GPT-2 automaticamente.

### Allenare il modello

Dopodiché avvia il training:

```bash
python train_transformer.py
```

Questo script:

- carica il tokenizer (quello personalizzato in `lib/tokenizer.json` se presente, altrimenti GPT-2)
- carica il dataset italiano (con barre di avanzamento su lettura e tokenizzazione)
- crea blocchi di token per l'addestramento, con cache su disco keyed per fingerprint dei dati e dimensione del vocabolario
- calcola la loss sulla predizione del token successivo
- salva i checkpoint in `runs/`, compreso `runs/best_model.safetensors` e `runs/final_model.safetensors`

### Generare testo

```bash
python app.py
```

Questo script carica il modello salvato e il tokenizer, ridimensiona gli embedding per includere i token speciali (se necessario) e genera testo a partire da un prompt usando top-k, top-p e repetition penalty.

## Piano futuro

- continuare l'addestramento RL (Reinforcement Learning) per migliorare la coerenza e lo stile
- aggiungere il Siciliano e insegnare al modello a parlare Siciliano stretto
- trasformare il progetto in un vero gioco didattico per imparare LLM e linguistica italiana

## Nota

Per ora il modello è un piccolo esperimento locale. Il focus è comprendere la pipeline di training, i dati e la generazione testuale con un LLM in italiano.
E ovviamente questo è un progetto ironico.