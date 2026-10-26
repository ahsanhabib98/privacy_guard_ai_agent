# Run PrivacyGuard

## 1. Go to the project folder

```bash
cd privacy_guard_ai_agent
```

## 2. Create and activate the virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Create the `.env` file

```bash
cp .env.example .env
```

Make sure `.env` contains a valid `PRIVACY_HMAC_SECRET`.

## 5. Make scripts executable

```bash
chmod +x scripts/*.sh
```

## 6. Pull the Qwen model

```bash
./scripts/pull_model.sh
```

## 7. Start Ollama

Open **Terminal 1**:

```bash
./scripts/run_ollama.sh
```

Keep it running.

## 8. Run PrivacyGuard

Open **Terminal 2**:

```bash
cd continuous_learning_AI_agent
source venv/bin/activate
```

### Check status

```bash
python -m app.cli status
```

### Process text

```bash
python -m app.cli process \
  --text "Contact Sarah Khan at sarah.khan@company.com." \
  --scope demo
```

### Check memory

```bash
python -m app.cli memory-stats
```

### Using the helper script

You can also use:

```bash
./scripts/run_cli.sh status
```

or:

```bash
./scripts/run_cli.sh process \
  --text "Contact Sarah Khan at sarah.khan@company.com." \
  --scope demo
```

### Using given sample input file

You can also use:

```bash
python -m app.cli process --file input.txt --scope demo 
```