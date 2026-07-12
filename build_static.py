#!/usr/bin/env python3
"""
Build static HTML from Jinja2 templates for Vercel deployment.
Renders all pages with a mock API_BASE_URL for static hosting.
"""
import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

# Setup Jinja2 environment
templates_dir = Path(__file__).parent / 'templates'
env = Environment(loader=FileSystemLoader(str(templates_dir)))

# Mock url_for for static files
def url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', '')
        return f'/static/{filename}'
    return '#'

# Mock config
class MockConfig:
    MAX_CONTENT_LENGTH = 64 * 1024 * 1024

config = MockConfig()

# Base context available to all templates
API_BASE_URL = os.environ.get('NEXT_PUBLIC_API_BASE_URL', '')
base_context = {
    'url_for': url_for,
    'config': config,
    'request': type('obj', (object,), {'path': '/'})(),
    'api_base_url': API_BASE_URL,
    'hostname': 'vercel.app' if 'vercel' in os.environ.get('VERCEL', '') else 'unknown',
}

# Pages to build: (template, output_file, extra_context)
pages = [
    ('index.html', 'index.html', {'active_page': 'home'}),
    ('camera.html', 'camera.html', {'active_page': 'camera'}),
    ('analysis.html', 'analysis.html', {'active_page': 'analysis'}),
    ('dashboard.html', 'dashboard.html', {'active_page': 'dashboard'}),
    ('reports.html', 'reports.html', {'active_page': 'reports'}),
    ('settings.html', 'settings.html', {'active_page': 'settings'}),
]

# Output directory
dist_dir = Path(__file__).parent / 'dist'
dist_dir.mkdir(exist_ok=True)

# Also copy static files
import shutil
static_src = Path(__file__).parent / 'static'
static_dst = dist_dir / 'static'
if static_dst.exists():
    shutil.rmtree(static_dst)
shutil.copytree(static_src, static_dst)

# Inject environment variables into api-config.js
api_config_path = static_dst / 'js' / 'api-config.js'
if api_config_path.exists():
    content = api_config_path.read_text()
    api_url = os.environ.get('NEXT_PUBLIC_API_BASE_URL', '')
    if api_url:
        content = content.replace("{{NEXT_PUBLIC_API_BASE_URL}}", api_url)
    api_config_path.write_text(content)

print("Building static pages...")
for template_name, output_name, extra_context in pages:
    template = env.get_template(template_name)
    context = {**base_context, **extra_context}
    html = template.render(**context)
    
    output_path = dist_dir / output_name
    output_path.write_text(html)
    print(f"  ✓ {output_name}")

# No _redirects needed — cleanUrls in vercel.json handles routing

print(f"\nBuild complete! Output in {dist_dir}/")
print(f"Deploy with: vercel --prod")