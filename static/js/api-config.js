// API Configuration - Auto-detects environment
// When deployed on Vercel, this will use the Vercel env var
// When running locally, falls back to localhost

(function() {
    // Check for environment variable (set at build time)
    const ENV_API_URL = '{{NEXT_PUBLIC_API_BASE_URL}}' || '';
    
    // Auto-detect based on hostname
    const hostname = window.location.hostname;
    let API_BASE_URL = '';
    
    if (ENV_API_URL && ENV_API_URL !== '{{NEXT_PUBLIC_API_BASE_URL}}') {
        // Build-time environment variable
        API_BASE_URL = ENV_API_URL;
    } else if (hostname === 'localhost' || hostname === '127.0.0.1') {
        // Local development
        API_BASE_URL = 'http://localhost:5050';
    } else if (hostname.includes('vercel.app') || hostname.includes('rice-analyzer')) {
        // Vercel deployment - use the Cloudflare tunnel
        API_BASE_URL = 'https://rice-api.yash-patel.in';
    } else {
        // Fallback
        API_BASE_URL = 'http://localhost:5050';
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