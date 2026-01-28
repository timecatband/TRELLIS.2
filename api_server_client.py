#!/usr/bin/env python3
"""
TRELLIS 2 API Server Client

A command-line client for the TRELLIS 2 API server that generates
untextured and textured GLB files from an image.
"""

import argparse
import base64
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("Error: requests library not found. Install with: pip install requests")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: PIL library not found. Install with: pip install pillow")
    sys.exit(1)


class TrellisAPIClient:
    def __init__(self, api_url: str = "http://localhost:8082", timeout: int = 600):
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        
    def healthcheck(self) -> bool:
        """Check if the API server is healthy"""
        try:
            response = requests.get(f"{self.api_url}/healthcheck", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def encode_image(self, image_path: str) -> str:
        """Encode image file to base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    
    def generate_shape(
        self,
        image_path: str,
        seed: Optional[int] = None,
        ss_guidance_scale: float = 7.5,
        ss_sampling_steps: int = 12,
        shape_guidance_scale: float = 7.5,
        shape_sampling_steps: int = 12,
    ) -> tuple[bytes, str]:
        """
        Generate untextured geometry.
        
        Returns:
            Tuple of (glb_bytes, image_hash)
        """
        print(f"📤 Uploading image: {image_path}")
        image_b64 = self.encode_image(image_path)
        
        payload = {
            'image': image_b64,
            'ss_guidance_scale': ss_guidance_scale,
            'ss_sampling_steps': ss_sampling_steps,
            'shape_guidance_scale': shape_guidance_scale,
            'shape_sampling_steps': shape_sampling_steps,
        }
        
        if seed is not None:
            payload['seed'] = seed
        
        print("🔨 Generating geometry (this may take a minute)...")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_msg = response.json().get('text', 'Unknown error')
                raise RuntimeError(f"Shape generation failed: {error_msg}")
            
            elapsed = time.time() - start_time
            print(f"✅ Geometry generated in {elapsed:.1f}s")
            
            image_hash = response.headers.get('X-Image-Hash')
            if not image_hash:
                print("⚠️  Warning: No image hash in response")
            
            return response.content, image_hash
            
        except requests.Timeout:
            raise RuntimeError("Shape generation timed out")
        except requests.RequestException as e:
            raise RuntimeError(f"Request failed: {e}")
    
    def check_status(self, image_hash: str) -> dict:
        """Check job status"""
        try:
            response = requests.get(f"{self.api_url}/status/{image_hash}", timeout=10)
            if response.status_code == 404:
                return {'status': 'not_found'}
            return response.json()
        except Exception as e:
            print(f"⚠️  Status check failed: {e}")
            return {'status': 'unknown'}
    
    def generate_texture(
        self,
        image_hash: str,
        tex_guidance_scale: float = 7.5,
        tex_sampling_steps: int = 12,
        poll_interval: int = 2,
    ) -> bytes:
        """
        Generate textured mesh.
        
        Args:
            image_hash: Hash from generate_shape response
            tex_guidance_scale: Texture guidance scale
            tex_sampling_steps: Texture sampling steps
            poll_interval: How often to poll status (seconds)
        
        Returns:
            GLB bytes
        """
        print("🎨 Generating texture (running in background)...")
        
        # Poll status while texture generates
        last_status = None
        dots = 0
        
        while True:
            status = self.check_status(image_hash)
            current_status = status.get('status')
            
            # Print status change
            if current_status != last_status:
                if current_status == 'texture_processing':
                    print("   Texture generation in progress...")
                elif current_status == 'completed':
                    print("   Texture generation completed!")
                    break
                elif current_status == 'error':
                    error = status.get('error', 'Unknown error')
                    raise RuntimeError(f"Texture generation failed: {error}")
                last_status = current_status
            else:
                # Print progress dots
                print("." * (dots % 3 + 1) + " " * (2 - dots % 3), end='\r', flush=True)
                dots += 1
            
            time.sleep(poll_interval)
        
        # Texture is ready, fetch it
        print("📥 Downloading textured mesh...")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.api_url}/generate_texture",
                json={'image_hash': image_hash},
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_msg = response.json().get('text', 'Unknown error')
                raise RuntimeError(f"Texture download failed: {error_msg}")
            
            elapsed = time.time() - start_time
            print(f"✅ Textured mesh downloaded in {elapsed:.1f}s")
            
            return response.content
            
        except requests.Timeout:
            raise RuntimeError("Texture download timed out")
        except requests.RequestException as e:
            raise RuntimeError(f"Request failed: {e}")
    
    def generate_full(
        self,
        image_path: str,
        output_shape: Optional[str] = None,
        output_textured: Optional[str] = None,
        seed: Optional[int] = None,
        ss_guidance_scale: float = 7.5,
        ss_sampling_steps: int = 12,
        shape_guidance_scale: float = 7.5,
        shape_sampling_steps: int = 12,
        tex_guidance_scale: float = 7.5,
        tex_sampling_steps: int = 12,
    ) -> tuple[str, str]:
        """
        Generate both untextured and textured meshes.
        
        Returns:
            Tuple of (shape_path, textured_path)
        """
        # Set default output paths
        if output_shape is None:
            stem = Path(image_path).stem
            output_shape = f"{stem}_shape.glb"
        
        if output_textured is None:
            stem = Path(image_path).stem
            output_textured = f"{stem}_textured.glb"
        
        # Generate shape
        shape_bytes, image_hash = self.generate_shape(
            image_path,
            seed=seed,
            ss_guidance_scale=ss_guidance_scale,
            ss_sampling_steps=ss_sampling_steps,
            shape_guidance_scale=shape_guidance_scale,
            shape_sampling_steps=shape_sampling_steps,
        )
        
        # Save untextured mesh
        with open(output_shape, 'wb') as f:
            f.write(shape_bytes)
        print(f"💾 Saved untextured mesh: {output_shape}")
        
        # Generate texture
        textured_bytes = self.generate_texture(
            image_hash,
            tex_guidance_scale=tex_guidance_scale,
            tex_sampling_steps=tex_sampling_steps,
        )
        
        # Save textured mesh
        with open(output_textured, 'wb') as f:
            f.write(textured_bytes)
        print(f"💾 Saved textured mesh: {output_textured}")
        
        return output_shape, output_textured


def main():
    parser = argparse.ArgumentParser(
        description="TRELLIS 2 API Client - Generate 3D meshes from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python api_server_client.py input.png
  
  # Custom output paths
  python api_server_client.py input.png --output-shape shape.glb --output-textured final.glb
  
  # Custom server and seed
  python api_server_client.py input.png --api-url http://192.168.1.100:8082 --seed 123
  
  # Only generate shape (no texture)
  python api_server_client.py input.png --shape-only
  
  # Adjust generation parameters
  python api_server_client.py input.png --ss-guidance 10.0 --shape-guidance 10.0
        """
    )
    
    # Required arguments
    parser.add_argument(
        'image',
        type=str,
        help='Input image path'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-shape',
        type=str,
        default=None,
        help='Output path for untextured mesh (default: <input>_shape.glb)'
    )
    parser.add_argument(
        '--output-textured',
        type=str,
        default=None,
        help='Output path for textured mesh (default: <input>_textured.glb)'
    )
    parser.add_argument(
        '--shape-only',
        action='store_true',
        help='Only generate shape, skip texture generation'
    )
    
    # Server arguments
    parser.add_argument(
        '--api-url',
        type=str,
        default='http://localhost:8082',
        help='API server URL (default: http://localhost:8082)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=600,
        help='Request timeout in seconds (default: 600)'
    )
    
    # Generation arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for generation (default: server default)'
    )
    parser.add_argument(
        '--ss-guidance',
        type=float,
        default=7.5,
        help='Sparse structure guidance scale (default: 7.5)'
    )
    parser.add_argument(
        '--ss-steps',
        type=int,
        default=12,
        help='Sparse structure sampling steps (default: 12)'
    )
    parser.add_argument(
        '--shape-guidance',
        type=float,
        default=7.5,
        help='Shape guidance scale (default: 7.5)'
    )
    parser.add_argument(
        '--shape-steps',
        type=int,
        default=12,
        help='Shape sampling steps (default: 12)'
    )
    parser.add_argument(
        '--tex-guidance',
        type=float,
        default=7.5,
        help='Texture guidance scale (default: 7.5)'
    )
    parser.add_argument(
        '--tex-steps',
        type=int,
        default=12,
        help='Texture sampling steps (default: 12)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.image).exists():
        print(f"❌ Error: Image file not found: {args.image}")
        sys.exit(1)
    
    try:
        # Verify it's a valid image
        Image.open(args.image)
    except Exception as e:
        print(f"❌ Error: Invalid image file: {e}")
        sys.exit(1)
    
    # Create client
    client = TrellisAPIClient(api_url=args.api_url, timeout=args.timeout)
    
    # Health check
    print(f"🔍 Checking API server at {args.api_url}...")
    if not client.healthcheck():
        print(f"❌ Error: API server not responding at {args.api_url}")
        print("   Make sure the server is running:")
        print("   python api_server_trellis2.py")
        sys.exit(1)
    print("✅ API server is healthy\n")
    
    # Generate
    print("=" * 60)
    print(f"TRELLIS 2 - Image to 3D Generation")
    print("=" * 60)
    print(f"Input: {args.image}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print()
    
    try:
        start_time = time.time()
        
        if args.shape_only:
            # Only generate shape
            shape_bytes, _ = client.generate_shape(
                args.image,
                seed=args.seed,
                ss_guidance_scale=args.ss_guidance,
                ss_sampling_steps=args.ss_steps,
                shape_guidance_scale=args.shape_guidance,
                shape_sampling_steps=args.shape_steps,
            )
            
            output_shape = args.output_shape or f"{Path(args.image).stem}_shape.glb"
            with open(output_shape, 'wb') as f:
                f.write(shape_bytes)
            print(f"💾 Saved untextured mesh: {output_shape}")
            
        else:
            # Generate both shape and texture
            output_shape, output_textured = client.generate_full(
                args.image,
                output_shape=args.output_shape,
                output_textured=args.output_textured,
                seed=args.seed,
                ss_guidance_scale=args.ss_guidance,
                ss_sampling_steps=args.ss_steps,
                shape_guidance_scale=args.shape_guidance,
                shape_sampling_steps=args.shape_steps,
                tex_guidance_scale=args.tex_guidance,
                tex_sampling_steps=args.tex_steps,
            )
        
        total_time = time.time() - start_time
        
        print()
        print("=" * 60)
        print(f"✨ Generation complete in {total_time:.1f}s!")
        print("=" * 60)
        
        if args.shape_only:
            print(f"Untextured mesh: {output_shape}")
        else:
            print(f"Untextured mesh: {output_shape}")
            print(f"Textured mesh:   {output_textured}")
        
    except RuntimeError as e:
        print()
        print("=" * 60)
        print(f"❌ Error: {e}")
        print("=" * 60)
        sys.exit(1)
    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("⚠️  Interrupted by user")
        print("=" * 60)
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ Unexpected error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
