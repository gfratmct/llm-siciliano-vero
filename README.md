# llm-pelatone

Un LLM trainato con dati italiani al 100% e RL (soon).

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

### Preparare i dati

1. Metti i file di testo in `data/` con estensione `.utf8`.
2. Assicurati che il contenuto sia italiano.

### Allenare il modello

Prima di eseguire l'allenamento, sincronizza le dipendenze col tool `uv`.

#### Installazione CPU-only

```bash
uv sync
```

#### Installazione con PyTorch CUDA (se hai GPU e vuoi usare CUDA 11.8)

```bash
uv sync --index pytorch-cuda
```

> Se `uv sync --index pytorch-cuda` non è disponibile o non è compatibile con il tuo sistema, usa `uv sync` per installare la versione CPU.

Dopodiché avvia il training:

```bash
python3 train.py
```

Questo script:

- carica il dataset italiano
- usa il tokenizer GPT-2 con `Tokenizer.from_pretrained("gpt2")`
- crea blocchi di token per l'addestramento
- calcola la loss sulla predizione del token successivo
- salva i checkpoint in `runs/`, compreso `runs/best_model.safetensors` e `runs/final_model.safetensors`

### Generare testo

```bash
python3 app.py
```

Questo script carica il modello salvato, carica il tokenizer GPT-2 e genera testo a partire da un prompt.

## Piano futuro

- continuare l'addestramento RL (Reinforcement Learning) per migliorare la coerenza e lo stile
- aggiungere il Siciliano e insegnare al modello a parlare Siciliano stretto
- trasformare il progetto in un vero gioco didattico per imparare LLM e linguistica italiana

## Nota

Per ora il modello è un piccolo esperimento locale. Il focus è comprendere la pipeline di training, i dati e la generazione testuale con un LLM in italiano.
