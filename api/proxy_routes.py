import requests
from flask import Blueprint, request, Response, stream_with_context
import logging
import os
import time

proxy_bp = Blueprint('proxy_routes', __name__, url_prefix='/proxy')
logger = logging.getLogger(__name__)

# Support Docker service names or localhost fallback.
# On Windows/non-docker deployments, defaulting to "mediamtx" causes DNS failures and 502s from the proxy.
MEDIAMTX_HOST = os.environ.get('MEDIAMTX_HOST', '127.0.0.1')
MEDIAMTX_HLS_URL = os.environ.get('MEDIAMTX_HLS_URL', f"http://{MEDIAMTX_HOST}:8888")
MEDIAMTX_WEBRTC_URL = os.environ.get('MEDIAMTX_WEBRTC_URL', f"http://{MEDIAMTX_HOST}:8889")

# Connection settings for reliability
REQUEST_TIMEOUT = 10  # seconds
MAX_RETRIES = 2
RETRY_DELAY = 0.5  # seconds

# Log configuration on startup
logger.info(f"Proxy configured with HLS={MEDIAMTX_HLS_URL}, WebRTC={MEDIAMTX_WEBRTC_URL}")

def _make_request_with_retry(method, url, headers, data, cookies, max_retries=MAX_RETRIES):
    """Make HTTP request with retry logic for transient failures."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                data=data,
                cookies=cookies,
                allow_redirects=False,
                stream=True,
                timeout=REQUEST_TIMEOUT
            )
            return resp
        except requests.exceptions.ConnectionError as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(RETRY_DELAY)
                continue
        except Exception as e:
            last_error = e
            break
    raise last_error

@proxy_bp.route('/hls/<path:path>', methods=['GET', 'OPTIONS'])
def proxy_hls(path):
    if request.method == 'OPTIONS':
        resp = Response('', 200)
        return resp

    url = f"{MEDIAMTX_HLS_URL}/{path}"
    try:
        headers_dict = {key: value for (key, value) in request.headers if key.lower() != 'host'}
        resp = _make_request_with_retry(
            method=request.method,
            url=url,
            headers=headers_dict,
            data=request.get_data(),
            cookies=request.cookies
        )
        
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        headers = [(name, value) for (name, value) in resp.headers.items()
                   if name.lower() not in excluded_headers]
        
        return Response(
            stream_with_context(resp.iter_content(chunk_size=8192)),
            status=resp.status_code,
            headers=headers
        )
    except Exception as e:
        logger.error(f"HLS Proxy error for {path}: {e}")
        return Response(f"Proxy error: {str(e)}", 502)

@proxy_bp.route('/webrtc/<path:path>', methods=['GET', 'POST', 'DELETE', 'PATCH', 'OPTIONS'])
def proxy_webrtc(path):
    if request.method == 'OPTIONS':
        resp = Response('', 200)
        return resp

    url = f"{MEDIAMTX_WEBRTC_URL}/{path}"
    # Pass query parameters like ?quality=high
    if request.query_string:
        url = f"{url}?{request.query_string.decode('utf-8')}"
    try:
        headers_dict = {key: value for (key, value) in request.headers if key.lower() != 'host'}
        
        # For WHEP/WHIP, we need to forward the body
        resp = _make_request_with_retry(
            method=request.method,
            url=url,
            headers=headers_dict,
            data=request.get_data(),
            cookies=request.cookies
        )
        
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        headers = [(name, value) for (name, value) in resp.headers.items()
                   if name.lower() not in excluded_headers]
        
        return Response(
            stream_with_context(resp.iter_content(chunk_size=8192)),
            status=resp.status_code,
            headers=headers
        )
    except Exception as e:
        logger.error(f"WebRTC Proxy error for {path}: {e}")
        return Response(f"Proxy error: {str(e)}", 502)

