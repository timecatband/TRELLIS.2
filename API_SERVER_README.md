# TRELLIS 2 Unified API Server

A unified FastAPI server for TRELLIS 2 geometry and texture generation, compatible with Hunyuan3D API endpoints.

## Features

- **Unified Server**: Single server handles both geometry and texture generation (unlike Hunyuan's separate servers)
- **Asynchronous Processing**: Returns untextured GLB immediately while texture generation continues in background
- **Job Tracking**: Uses image MD5 hash to match geometry and texture requests
- **Compatible API**: Drop-in replacement for Hunyuan3D API endpoints

## Architecture

```
Client Request (image) → /generate → Untextured GLB (immediate)
                                   ↓
                         Texture Generation (background thread)
                                   ↓
Client Request (hash) → /generate_texture → Textured GLB (blocks until ready)
```

## Installation

```bash
# Install TRELLIS 2 dependencies
cd TRELLIS.2
pip install -r requirements.txt

# Install API server dependencies
pip install fastapi uvicorn
```

## Usage

### Configuration

The server is configured using a JSON file. Create your own config or use the default:

```bash
# Use default config
python api_server_trellis2.py

# Use custom config
python api_server_trellis2.py --config my_config.json
```

### Configuration File Format

See `api_server_config_default.json` for the default configuration:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8082,
    "log_level": "info"
  },
  "model": {
    "model_path": "microsoft/TRELLIS.2-4B",
    "device": "cuda"
  },
  "generation": {
    "seed": 42,
    "pipeline_type": "1024_cascade",
    "max_num_tokens": 49152
  },
  "export": {
    "glb_decimation_target": 1000000,
    "glb_texture_size": 4096
  },
  "worker": {
    "limit_model_concurrency": 5,
    "job_cleanup_interval_seconds": 300,
    "job_max_age_seconds": 3600
  }
}
```

### Configuration Parameters

#### Server Section
- `host`: Server host (default: 0.0.0.0)
- `port`: Server port (default: 8082)
- `log_level`: Logging level (default: info)

#### Model Section
- `model_path`: Path to TRELLIS 2 model (default: microsoft/TRELLIS.2-4B)
- `device`: Device for inference (default: cuda)

#### Generation Section
- `seed`: Random seed (default: 42)
- `pipeline_type`: Pipeline resolution mode (default: 1024_cascade)
  - `512`: Fast, lower quality
  - `1024`: High quality, more VRAM
  - `1024_cascade`: Best quality/VRAM balance
  - `1536_cascade`: Highest quality, most VRAM
- `max_num_tokens`: Max tokens for cascade modes (default: 49152)
  - Lower = less VRAM but potentially lower quality
  - Try 32768 (~7GB) or 24576 (~6GB) for lower VRAM

#### Export Section
- `glb_decimation_target`: Target face count for GLB export (default: 1,000,000)
- `glb_texture_size`: Texture resolution for GLB export (default: 4096)

#### Worker Section
- `limit_model_concurrency`: Max concurrent requests (default: 5)
- `job_cleanup_interval_seconds`: How often to clean up old jobs (default: 300)
- `job_max_age_seconds`: Max age of jobs before cleanup (default: 3600)

### Start Server

```bash
# With default config
python api_server_trellis2.py

# With custom config
python api_server_trellis2.py --config my_config.json

# With low VRAM config
python api_server_trellis2.py --config api_server_config_low_vram.json

# With high quality config
python api_server_trellis2.py --config api_server_config_high_quality.json
```

### Example Configurations

Three example configurations are provided:

1. **`api_server_config_default.json`** - Balanced quality and performance
   - Pipeline: 1024_cascade
   - Good for most use cases
   - ~9GB VRAM

2. **`api_server_config_low_vram.json`** - Optimized for lower VRAM
   - Pipeline: 512
   - Reduced texture size and decimation
   - ~4-5GB VRAM

3. **`api_server_config_high_quality.json`** - Maximum quality
   - Pipeline: 1536_cascade
   - Higher resolution textures
   - ~12-16GB VRAM
   - Reduced concurrency for stability

## API Endpoints

### POST /generate

Generate geometry (untextured mesh). Returns GLB immediately while texture generation starts in background.

**Request:**
```json
{
  "image": "base64_encoded_image",
  "seed": 42,
  "ss_guidance_scale": 7.5,
  "ss_sampling_steps": 12,
  "shape_guidance_scale": 7.5,
  "shape_sampling_steps": 12,
  "tex_guidance_scale": 7.5,
  "tex_sampling_steps": 12
}
```

**Response:**
- Binary GLB file (untextured mesh)
- Header: `X-Image-Hash` - Use this for `/generate_texture` request

**Example:**
```python
import base64
import requests
from PIL import Image

# Load and encode image
with open("input.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Request geometry
response = requests.post(
    "http://localhost:8082/generate",
    json={
        "image": image_b64,
        "seed": 42,
        "ss_guidance_scale": 7.5,
        "ss_sampling_steps": 12,
        "shape_guidance_scale": 7.5,
        "shape_sampling_steps": 12,
    }
)

# Save untextured mesh
with open("output_shape.glb", "wb") as f:
    f.write(response.content)

# Get image hash for texture request
image_hash = response.headers.get("X-Image-Hash")
```

### POST /generate_texture

Generate texture for previously generated geometry. Blocks until texture is ready.

**Request (Option 1 - Using image hash):**
```json
{
  "image_hash": "abc123..."
}
```

**Request (Option 2 - Using same image):**
```json
{
  "image": "base64_encoded_image"
}
```

**Response:**
- Binary GLB file (textured mesh with PBR materials)

**Example:**
```python
# Request texture (blocks until ready)
response = requests.post(
    "http://localhost:8082/generate_texture",
    json={"image_hash": image_hash}
)

# Save textured mesh
with open("output_textured.glb", "wb") as f:
    f.write(response.content)
```

### GET /status/{image_hash}

Check generation status for a job.

**Response:**
```json
{
  "status": "shape_processing" | "texture_processing" | "completed" | "error",
  "uid": "job_uuid",
  "error": "error message (if status=error)"
}
```

### GET /healthcheck

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

### GET /worker_status

Get worker queue status.

**Response:**
```json
{
  "speed": 1,
  "queue_length": 0
}
```

## Complete Example

```python
import base64
import requests
import time
from PIL import Image

API_URL = "http://localhost:8082"

# 1. Load image
with open("input.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# 2. Generate geometry (returns immediately)
print("Generating geometry...")
response = requests.post(
    f"{API_URL}/generate",
    json={
        "image": image_b64,
        "seed": 42,
        "ss_guidance_scale": 7.5,
        "ss_sampling_steps": 12,
        "shape_guidance_scale": 7.5,
        "shape_sampling_steps": 12,
    }
)

# 3. Save untextured mesh
with open("output_shape.glb", "wb") as f:
    f.write(response.content)
print("Untextured mesh saved!")

# 4. Get image hash
image_hash = response.headers.get("X-Image-Hash")

# 5. Optional: Check status while texture generates
while True:
    status = requests.get(f"{API_URL}/status/{image_hash}").json()
    print(f"Status: {status['status']}")
    if status['status'] in ('completed', 'error'):
        break
    time.sleep(2)

# 6. Get textured mesh (blocks until ready)
print("Waiting for texture...")
response = requests.post(
    f"{API_URL}/generate_texture",
    json={"image_hash": image_hash}
)

# 7. Save textured mesh
with open("output_textured.glb", "wb") as f:
    f.write(response.content)
print("Textured mesh saved!")
```

## Error Handling

The server returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (missing parameters, invalid input)
- `404`: Job not found
- `408`: Timeout (texture generation took too long)
- `500`: Server error

Error responses include JSON:
```json
{
  "text": "Error message",
  "error_code": 1
}
```

## Job Cleanup

Old jobs (>1 hour) are automatically cleaned up every 5 minutes to free memory.

## Memory Management

The server automatically:
- Clears CUDA cache between requests
- Removes intermediate data after texture completion
- Runs garbage collection

For lower VRAM usage, adjust your config file:
- Set `pipeline_type` to `512` (fastest, lowest VRAM)
- Reduce `max_num_tokens` (e.g., 32768 or 24576)
- Reduce `glb_texture_size` (e.g., 2048)
- Reduce `glb_decimation_target` (e.g., 500000)

Example low-VRAM config:
```json
{
  "generation": {
    "pipeline_type": "512",
    "max_num_tokens": 24576
  },
  "export": {
    "glb_decimation_target": 500000,
    "glb_texture_size": 2048
  }
}
```

## Differences from Hunyuan3D API

1. **Unified Server**: Single server instead of separate geometry/texture servers
2. **Asynchronous**: Geometry returns immediately, texture in background
3. **Job Tracking**: Uses image hash to match requests
4. **Compatible**: Same endpoint names and request format

## Notes

- The server uses the same image for both geometry and texture generation
- Image hash is computed from the preprocessed image (after background removal)
- Texture generation starts automatically after geometry generation
- The `/generate_texture` endpoint will block until texture is ready (default timeout: 5 minutes)
