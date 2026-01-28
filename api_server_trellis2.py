#!/usr/bin/env python3
"""
TRELLIS 2 Unified API Server

Handles both geometry and texture generation in a single server.
Compatible with Hunyuan3D API endpoints.

Key features:
- /generate: Returns untextured GLB immediately, starts texture generation in background
- /generate_texture: Blocks until texture is ready
- Jobs tracked by image MD5 hash for matching geometry/texture requests
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import asyncio
import base64
import gc
import hashlib
import json
import logging
import logging.handlers
import signal
import sys
import tempfile
import threading
import time
import traceback
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import trimesh
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# TRELLIS 2 imports
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

LOGDIR = '.'
SAVE_DIR = 'trellis_cache'
os.makedirs(SAVE_DIR, exist_ok=True)

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger("trellis_controller", f"{SAVE_DIR}/trellis_controller.log")

# Global variables
model_semaphore = None


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate required sections
    required_sections = ['server', 'model', 'generation', 'export', 'worker']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    return config


def compute_image_hash(image: Image.Image) -> str:
    """Compute MD5 hash of image for job tracking"""
    img_bytes = BytesIO()
    image.save(img_bytes, format='PNG')
    return hashlib.md5(img_bytes.getvalue()).hexdigest()


class JobTracker:
    """Track jobs by image hash for matching geometry/texture requests"""
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def create_job(self, image_hash: str, uid: str) -> None:
        """Create a new job entry"""
        with self.lock:
            self.jobs[image_hash] = {
                'uid': uid,
                'status': 'shape_processing',
                'shape_path': None,
                'texture_path': None,
                'shape_result': None,  # Store intermediate data for texture generation
                'conditioning': None,
                'error': None,
                'created_at': time.time(),
            }
    
    def update_job(self, image_hash: str, **kwargs) -> None:
        """Update job status"""
        with self.lock:
            if image_hash in self.jobs:
                self.jobs[image_hash].update(kwargs)
    
    def get_job(self, image_hash: str) -> Optional[Dict[str, Any]]:
        """Get job info"""
        with self.lock:
            return self.jobs.get(image_hash)
    
    def cleanup_old_jobs(self, max_age_seconds: int = 3600):
        """Remove jobs older than max_age_seconds"""
        with self.lock:
            now = time.time()
            to_remove = [
                h for h, job in self.jobs.items()
                if now - job['created_at'] > max_age_seconds
            ]
            for h in to_remove:
                del self.jobs[h]


class Trellis2Worker:
    def __init__(self,
                 model_path: str = 'microsoft/TRELLIS.2-4B',
                 device: str = 'cuda',
                 seed: int = 42,
                 limit_model_concurrency: int = 5,
                 pipeline_type: str = '1024_cascade',
                 max_num_tokens: int = 49152,
                 glb_decimation_target: int = 1000000,
                 glb_texture_size: int = 4096):
        self.worker_id = worker_id
        self.device = device
        self.seed = seed
        self.limit_model_concurrency = limit_model_concurrency
        self.pipeline_type = pipeline_type
        self.max_num_tokens = max_num_tokens
        self.glb_decimation_target = glb_decimation_target
        self.glb_texture_size = glb_texture_size
        
        # Job tracking
        self.job_tracker = JobTracker()
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        logger.info(f"Loading TRELLIS 2 model on worker {worker_id} ...")
        
        # Load pipeline
        total_start_time = time.time()
        logger.info('Model loading started')
        
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
        self.pipeline.to(torch.device(device))
        
        logger.info(f'Total model loading time: {time.time() - total_start_time:.2f} seconds')

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return self.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_shape(self, uid: str, params: dict) -> tuple[str, str]:
        """
        Generate shape (geometry only).
        Returns (shape_glb_path, image_hash).
        Starts texture generation in background.
        """
        # Load image
        if 'image' in params:
            image = params["image"]
            image = load_image_from_base64(image)
        else:
            raise ValueError("No image provided")
        
        # Compute image hash for job tracking
        image_hash = compute_image_hash(image)
        
        # Create job
        self.job_tracker.create_job(image_hash, uid)
        
        try:
            # Preprocess image
            image = self.pipeline.preprocess_image(image)
            
            # Get parameters
            seed = params.get("seed", self.seed)
            ss_guidance_scale = params.get('ss_guidance_scale', 7.5)
            ss_sampling_steps = params.get('ss_sampling_steps', 12)
            shape_guidance_scale = params.get('shape_guidance_scale', 7.5)
            shape_sampling_steps = params.get('shape_sampling_steps', 12)
            
            logger.info(f"[{uid}] Generating shape with seed={seed}")
            
            # Get conditioning
            torch.manual_seed(seed)
            cond_512 = self.pipeline.get_cond([image], 512)
            cond_1024 = None
            if self.pipeline_type != '512':
                cond_1024 = self.pipeline.get_cond([image], 1024)
            
            # Combine conditioning
            conditioning = {
                'cond_512': cond_512['cond'],
                'neg_cond': cond_512['neg_cond'],
            }
            if cond_1024 is not None:
                conditioning['cond_1024'] = cond_1024['cond']
            
            # Sample sparse structure
            ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[self.pipeline_type]
            coords = self.pipeline.sample_sparse_structure(
                cond_512, ss_res,
                num_samples=1,
                sampler_params={
                    'steps': ss_sampling_steps,
                    'guidance_strength': ss_guidance_scale,
                }
            )
            
            # Sample shape
            cond_dict_512 = {'cond': conditioning['cond_512'], 'neg_cond': conditioning['neg_cond']}
            cond_dict_1024 = None
            if 'cond_1024' in conditioning:
                cond_dict_1024 = {'cond': conditioning['cond_1024'], 'neg_cond': conditioning['neg_cond']}
            
            if self.pipeline_type == '512':
                shape_slat = self.pipeline.sample_shape_slat(
                    cond_dict_512, self.pipeline.models['shape_slat_flow_model_512'],
                    coords,
                    sampler_params={
                        'steps': shape_sampling_steps,
                        'guidance_strength': shape_guidance_scale,
                    }
                )
                res = 512
            elif self.pipeline_type == '1024':
                shape_slat = self.pipeline.sample_shape_slat(
                    cond_dict_1024, self.pipeline.models['shape_slat_flow_model_1024'],
                    coords,
                    sampler_params={
                        'steps': shape_sampling_steps,
                        'guidance_strength': shape_guidance_scale,
                    }
                )
                res = 1024
            elif self.pipeline_type in ('1024_cascade', '1536_cascade'):
                target_res = 1024 if self.pipeline_type == '1024_cascade' else 1536
                shape_slat, res = self.pipeline.sample_shape_slat_cascade(
                    cond_dict_512, cond_dict_1024,
                    self.pipeline.models['shape_slat_flow_model_512'],
                    self.pipeline.models['shape_slat_flow_model_1024'],
                    512, target_res,
                    coords,
                    sampler_params={
                        'steps': shape_sampling_steps,
                        'guidance_strength': shape_guidance_scale,
                    },
                    max_num_tokens=self.max_num_tokens
                )
            else:
                raise ValueError(f"Invalid pipeline type: {self.pipeline_type}")
            
            # Decode shape
            meshes, subs = self.pipeline.decode_shape_slat(shape_slat, res)
            mesh = meshes[0]
            mesh.fill_holes()
            
            # Export untextured mesh as GLB
            shape_output_path = os.path.join(SAVE_DIR, f'{uid}_shape.glb')
            
            # Simple GLB export for untextured mesh
            tri_mesh = trimesh.Trimesh(
                vertices=mesh.vertices.cpu().numpy(),
                faces=mesh.faces.cpu().numpy(),
                process=False
            )
            tri_mesh.export(shape_output_path)
            
            logger.info(f"[{uid}] Shape generated and saved to {shape_output_path}")
            
            # Store intermediate data for texture generation
            self.job_tracker.update_job(
                image_hash,
                status='texture_processing',
                shape_path=shape_output_path,
                shape_slat=shape_slat,
                subs=subs,
                meshes=meshes,
                resolution=res,
                conditioning=conditioning,
            )
            
            # Start texture generation in background
            threading.Thread(
                target=self._generate_texture_background,
                args=(uid, image_hash, seed, params),
                daemon=True
            ).start()
            
            return shape_output_path, image_hash
            
        except Exception as e:
            logger.error(f"[{uid}] Shape generation failed: {e}")
            traceback.print_exc()
            self.job_tracker.update_job(image_hash, status='error', error=str(e))
            raise

    def _generate_texture_background(self, uid: str, image_hash: str, seed: int, params: dict):
        """Generate texture in background thread"""
        try:
            job = self.job_tracker.get_job(image_hash)
            if not job:
                logger.error(f"[{uid}] Job not found for texture generation")
                return
            
            shape_slat = job['shape_slat']
            subs = job['subs']
            meshes = job['meshes']
            res = job['resolution']
            conditioning = job['conditioning']
            
            # Get texture parameters
            tex_guidance_scale = params.get('tex_guidance_scale', 7.5)
            tex_sampling_steps = params.get('tex_sampling_steps', 12)
            
            logger.info(f"[{uid}] Generating texture in background")
            
            # Sample texture
            torch.manual_seed(seed)
            
            # Select appropriate conditioning and model based on resolution
            if self.pipeline_type == '512':
                cond_dict = {'cond': conditioning['cond_512'], 'neg_cond': conditioning['neg_cond']}
                tex_slat = self.pipeline.sample_tex_slat(
                    cond_dict,
                    self.pipeline.models['tex_slat_flow_model_512'],
                    shape_slat,
                    sampler_params={
                        'steps': tex_sampling_steps,
                        'guidance_strength': tex_guidance_scale,
                    }
                )
            else:
                cond_dict = {'cond': conditioning['cond_1024'], 'neg_cond': conditioning['neg_cond']}
                tex_slat = self.pipeline.sample_tex_slat(
                    cond_dict,
                    self.pipeline.models['tex_slat_flow_model_1024'],
                    shape_slat,
                    sampler_params={
                        'steps': tex_sampling_steps,
                        'guidance_strength': tex_guidance_scale,
                    }
                )
            
            # Decode texture
            tex_voxels = self.pipeline.decode_tex_slat(tex_slat, subs)
            
            # Combine mesh with texture voxels
            mesh = meshes[0]
            voxel = tex_voxels[0]
            
            from trellis2.representations import MeshWithVoxel
            textured_mesh = MeshWithVoxel(
                mesh.vertices, mesh.faces,
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1 / res,
                coords=voxel.coords[:, 1:],
                attrs=voxel.feats,
                voxel_shape=torch.Size([*voxel.shape, *voxel.spatial_shape]),
                layout=self.pipeline.pbr_attr_layout
            )
            
            # Simplify
            textured_mesh.simplify(16777216)  # nvdiffrast limit
            
            # Export to GLB
            texture_output_path = os.path.join(SAVE_DIR, f'{uid}_textured.glb')
            
            glb = o_voxel.postprocess.to_glb(
                vertices=textured_mesh.vertices,
                faces=textured_mesh.faces,
                attr_volume=textured_mesh.attrs,
                coords=textured_mesh.coords,
                attr_layout=textured_mesh.layout,
                voxel_size=textured_mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=self.glb_decimation_target,
                texture_size=self.glb_texture_size,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=True
            )
            glb.export(texture_output_path, extension_webp=True)
            
            logger.info(f"[{uid}] Texture generated and saved to {texture_output_path}")
            
            # Update job
            self.job_tracker.update_job(
                image_hash,
                status='completed',
                texture_path=texture_output_path,
            )
            
            # Cleanup intermediate data to free memory
            self.job_tracker.update_job(
                image_hash,
                shape_slat=None,
                subs=None,
                meshes=None,
                conditioning=None,
            )
            
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"[{uid}] Texture generation failed: {e}")
            traceback.print_exc()
            self.job_tracker.update_job(image_hash, status='error', error=str(e))

    def wait_for_texture(self, image_hash: str, timeout: float = 300) -> Optional[str]:
        """Wait for texture generation to complete, return path or None"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            job = self.job_tracker.get_job(image_hash)
            if not job:
                return None
            
            if job['status'] == 'completed' and job['texture_path']:
                return job['texture_path']
            elif job['status'] == 'error':
                raise ValueError(f"Texture generation failed: {job['error']}")
            
            time.sleep(1)
        
        raise TimeoutError("Texture generation timed out")


# FastAPI app setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate")
async def generate(request: Request):
    """Generate geometry (shape only). Returns untextured GLB immediately."""
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Worker generating shape...")
    params = await request.json()
    uid = str(uuid.uuid4())
    
    try:
        shape_path, image_hash = worker.generate_shape(uid, params)
        logger.info(f"Generated shape saved at {shape_path}")
        
        # Return shape GLB immediately
        return FileResponse(
            shape_path,
            media_type="model/gltf-binary",
            headers={"X-Image-Hash": image_hash}  # Include hash for texture request
        )
    except ValueError as e:
        traceback.print_exc()
        logger.error(f"Caught ValueError: {e}")
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=400)
    except torch.cuda.CudaError as e:
        logger.error(f"Caught torch.cuda.CudaError: {e}")
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=500)
    except Exception as e:
        logger.error(f"Caught Unknown Error: {e}")
        traceback.print_exc()
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=500)


@app.post("/generate_texture")
async def generate_texture(request: Request):
    """
    Generate texture for previously generated geometry.
    Requires 'image' (same as used for /generate) or 'image_hash' from /generate response.
    Blocks until texture is ready.
    """
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Worker waiting for texture...")
    params = await request.json()
    
    try:
        # Get image hash
        if 'image_hash' in params:
            image_hash = params['image_hash']
        elif 'image' in params:
            image = load_image_from_base64(params['image'])
            image_hash = compute_image_hash(image)
        else:
            raise ValueError("Must provide 'image' or 'image_hash'")
        
        # Wait for texture generation to complete
        texture_path = worker.wait_for_texture(image_hash, timeout=300)
        
        if not texture_path or not os.path.exists(texture_path):
            raise ValueError("Texture generation failed or timed out")
        
        logger.info(f"Returning textured mesh from {texture_path}")
        
        return FileResponse(
            texture_path,
            media_type="model/gltf-binary"
        )
        
    except ValueError as e:
        traceback.print_exc()
        logger.error(f"Caught ValueError: {e}")
        ret = {
            "text": str(e),
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=400)
    except TimeoutError as e:
        logger.error(f"Caught TimeoutError: {e}")
        ret = {
            "text": str(e),
            "error_code": 2,
        }
        return JSONResponse(ret, status_code=408)
    except Exception as e:
        logger.error(f"Caught Unknown Error: {e}")
        traceback.print_exc()
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=500)


@app.get("/status/{image_hash}")
async def status(image_hash: str):
    """Check status of a job by image hash"""
    job = worker.job_tracker.get_job(image_hash)
    if not job:
        return JSONResponse({"status": "not_found"}, status_code=404)
    
    response = {
        "status": job['status'],
        "uid": job['uid'],
    }
    
    if job['status'] == 'error':
        response['error'] = job['error']
    
    return JSONResponse(response, status_code=200)


@app.get("/healthcheck")
async def healthcheck():
    """Health check endpoint"""
    return JSONResponse({"status": "ok"}, status_code=200)


@app.get("/worker_status")
async def worker_status():
    """Get worker status"""
    return JSONResponse(worker.get_status(), status_code=200)


@app.post("/quit")
async def quit():
    """Quit the server"""
    pid = os.getpid()
    os.kill(pid, signal.SIGKILL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TRELLIS 2 Unified API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python api_server_trellis2.py
  python api_server_trellis2.py --config my_config.json
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="api_server_config_default.json",
        help="Path to JSON config file (default: api_server_config_default.json)"
    )
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)
    logger.info(f"Config loaded: {json.dumps(config, indent=2)}")

    # Extract config values
    server_config = config['server']
    model_config = config['model']
    generation_config = config['generation']
    export_config = config['export']
    worker_config = config['worker']

    model_semaphore = asyncio.Semaphore(worker_config['limit_model_concurrency'])

    worker = Trellis2Worker(
        model_path=model_config['model_path'],
        device=model_config['device'],
        seed=generation_config['seed'],
        limit_model_concurrency=worker_config['limit_model_concurrency'],
        pipeline_type=generation_config['pipeline_type'],
        max_num_tokens=generation_config['max_num_tokens'],
        glb_decimation_target=export_config['glb_decimation_target'],
        glb_texture_size=export_config['glb_texture_size'],
    )
    
    # Start periodic cleanup of old jobs
    def cleanup_loop():
        while True:
            time.sleep(worker_config['job_cleanup_interval_seconds'])
            worker.job_tracker.cleanup_old_jobs(
                max_age_seconds=worker_config['job_max_age_seconds']
            )
    
    threading.Thread(target=cleanup_loop, daemon=True).start()
    
    uvicorn.run(
        app,
        host=server_config['host'],
        port=server_config['port'],
        log_level=server_config.get('log_level', 'info')
    )
