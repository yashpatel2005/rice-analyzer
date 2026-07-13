// API Configuration - resolves correct backend URL for every deployment scenario
// Scenarios:
//  1) Cloudflare Pages (frontend) + Cloudflare Tunnel backend (api.yash-patel.in)
//  2) Vercel frontend + backend on backend host
//  3) Flask serving both frontend+backend directly (same-origin)
//  4) Local development

(function() {
    const hostname = window.location.hostname;
    const protocol = window.location.protocol;
    let API_BASE_URL = '';

    // Priority 1: Explicit meta tag injected by backend/template
    const metaApiUrl = document.querySelector('meta[name="api-base-url"]');
    if (metaApiUrl && metaApiUrl.content) {
        API_BASE_URL = metaApiUrl.content.trim().replace(/\/$/, '');
    }
    // Priority 2: Cloudflare Pages or tunnel on yash-patel.in
    else if (
        hostname.includes('yash-patel.in') ||
        hostname.includes('rice-api') ||
        hostname.includes('pages.dev')
    ) {
        API_BASE_URL = 'https://api.yash-patel.in';
    }
    // Priority 3: Vercel deployment
    else if (hostname.includes('vercel.app')) {
        API_BASE_URL = 'https://api.yash-patel.in';
    }
    // Priority 4: Local development
    else if (hostname === 'localhost' || hostname === '127.0.0.1') {
        API_BASE_URL = 'http://localhost:5050';
    }
    // Priority 5: Same origin fallback (backend serves frontend)
    else {
        API_BASE_URL = '';
    }

    // Expose globally
    window.API_BASE_URL = API_BASE_URL;
    window.API_ENDPOINTS = {
        camera: {
            info: `${API_BASE_URL}/api/camera/info`,
            preview: `${API_BASE_URL}/api/camera/preview`,
            capture: `${API_BASE_URL}/api/camera/capture`,
            calibrate: `${API_BASE_URL}/api/camera/calibrate`,
            calibrateManual: `${API_BASE_URL}/api/camera/calibrate/manual`,
            calibrationStatus: `${API_BASE_URL}/api/camera/calibration_status`
        },
        analyze: {
            upload: `${API_BASE_URL}/api/analyze`,
            captured: `${API_BASE_URL}/api/analyze/captured`
        },
        reports: {
            list: `${API_BASE_URL}/api/reports`,
            latest: `${API_BASE_URL}/api/reports/latest`,
            runs: `${API_BASE_URL}/api/reports/list`,
            byId: (runId) => `${API_BASE_URL}/api/reports/${runId}`,
            download: (cat, file) => `${API_BASE_URL}/api/download/${cat}/${file}`,
            preview: (cat, file) => `${API_BASE_URL}/api/preview/${cat}/${file}`
        },
        dashboard: {
            data: `${API_BASE_URL}/api/dashboard/data`,
            exportPowerBI: `${API_BASE_URL}/api/export/powerbi`
        },
        settings: {
            get: `${API_BASE_URL}/api/settings`,
            update: `${API_BASE_URL}/api/settings`
        }
    };

    console.log('[API Config] Base URL:', API_BASE_URL);
})();
