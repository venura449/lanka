# lanka

## Engine health REST predictor

This server loads the trained model weights in `Model/`:

- `Model/engine_health_model.pkl`
- `Model/label_encoder.pkl`
- `Model/features.pkl`

Send sensor values via REST and read back the predicted status.

### Run

```bash
python -m pip install -r requirements.txt
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### REST endpoints

- `GET /health`: basic health check
- `POST /predict`: stateless prediction from a JSON payload
- `POST /telemetry`: stores latest values + returns prediction (microcontroller can call this)
- `GET /status`: returns latest stored inputs + prediction
  - returns `503` until `/telemetry` has been called at least once

### Payload format

```json
{
  "coolant_c": 102.5,
  "oil_psi": 35.2,
  "map_kpa": 42.1,
  "rpm": 850
}
```

### Model location (env vars)

- `MODEL_DIR` (default `./Model`)