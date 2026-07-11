/* ================================================================
   Rice Grain Morphometric Analysis System
   Main JavaScript — shared utilities v2.0
   ================================================================ */

// ------------------------------------------------------------------
// API Base URL (from api-config.js)
// ------------------------------------------------------------------
const API_BASE = window.API_BASE_URL || '';

// Helper for API calls
async function apiFetch(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const response = await fetch(url, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options.headers
        }
    });
    return response.json();
}

// ------------------------------------------------------------------
// Sidebar toggle (mobile)
// ------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebarOverlay');
    const toggleMobile = document.getElementById('sidebarToggleMobile');
    const toggle = document.getElementById('sidebarToggle');

    if (toggleMobile) {
        toggleMobile.addEventListener('click', () => {
            sidebar.classList.toggle('open');
            overlay.classList.toggle('active');
        });
    }
    if (toggle) {
        toggle.addEventListener('click', () => {
            sidebar.classList.toggle('open');
            overlay.classList.toggle('active');
        });
    }
    if (overlay) {
        overlay.addEventListener('click', () => {
            sidebar.classList.remove('open');
            overlay.classList.remove('active');
        });
    }

    // Highlight active nav item
    const path = window.location.pathname;
    document.querySelectorAll('.nav-item').forEach(a => {
        const href = a.getAttribute('href');
        if (href === path || (href !== '/' && path.startsWith(href))) {
            a.classList.add('active');
        } else {
            a.classList.remove('active');
        }
    });

    // Check calibration status (does NOT power on the camera)
    apiFetch('/api/camera/calibration_status')
        .then(data => {
            const el = document.getElementById('calibrationStatus');
            if (!el) return;
            if (data.success && data.is_calibrated) {
                el.innerHTML = `<i class="bi bi-check-circle-fill text-success"></i><span>Calibrated: ${data.pixels_per_mm.toFixed(2)} px/mm</span>`;
                el.style.borderColor = 'rgba(0,200,83,0.3)';
            } else if (data.success) {
                el.innerHTML = `<i class="bi bi-exclamation-triangle text-warning"></i><span>Not calibrated</span>`;
            } else {
                el.innerHTML = `<i class="bi bi-x-circle text-danger"></i><span>Status unavailable</span>`;
            }
        })
        .catch(() => {
            const el = document.getElementById('calibrationStatus');
            if (el) el.innerHTML = `<i class="bi bi-x-circle text-danger"></i><span>Status unavailable</span>`;
        });
});

// ------------------------------------------------------------------
// Spinner
// ------------------------------------------------------------------
function showSpinner(text) {
    const overlay = document.getElementById('loadingOverlay');
    const loadingText = document.getElementById('loadingText');
    if (loadingText) loadingText.textContent = text || 'Processing...';
    if (overlay) overlay.classList.add('active');
}

function hideSpinner() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.classList.remove('active');
}

// ------------------------------------------------------------------
// Toast notifications
// ------------------------------------------------------------------
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const iconMap = {
        success: 'bi-check-circle-fill text-success',
        error:   'bi-x-circle-fill text-danger',
        danger:  'bi-x-circle-fill text-danger',
        warning: 'bi-exclamation-triangle-fill text-warning',
        info:    'bi-info-circle-fill text-info',
    };

    const toast = document.createElement('div');
    toast.className = `toast toast-${type} show`;
    toast.innerHTML = `<i class="bi ${iconMap[type] || iconMap.info}"></i><span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(110%)';
        toast.style.transition = 'all 0.35s ease';
        setTimeout(() => toast.remove(), 350);
    }, 4000);
}

// ------------------------------------------------------------------
// Animated counter (rolls up numbers on load)
// ------------------------------------------------------------------
function animateCounter(el, target, duration = 900, decimals = 0) {
    if (!el) return;
    const start = 0;
    const startTime = performance.now();
    function update(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = start + (target - start) * eased;
        el.textContent = decimals > 0 ? current.toFixed(decimals) : Math.round(current).toString();
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// ------------------------------------------------------------------
// Image modal (fullscreen zoom viewer)
// ------------------------------------------------------------------
function openImgModal(src) {
    const backdrop = document.getElementById('imgModalBackdrop');
    const img = document.getElementById('imgModalImg');
    if (!backdrop || !img) return;
    img.src = src;
    backdrop.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeImgModal(event, force = false) {
    const backdrop = document.getElementById('imgModalBackdrop');
    if (!backdrop) return;
    if (force || (event && event.target === backdrop)) {
        backdrop.classList.remove('active');
        document.body.style.overflow = '';
        setTimeout(() => {
            document.getElementById('imgModalImg').src = '';
        }, 300);
    }
}

// Close on Escape key
document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeImgModal(null, true);
});

// ------------------------------------------------------------------
// noUiSlider helper — create a dual-handle slider
// ------------------------------------------------------------------
function createDualSlider(containerId, min, max, onUpdate) {
    const container = document.getElementById(containerId);
    if (!container || typeof noUiSlider === 'undefined') return null;
    if (container.noUiSlider) container.noUiSlider.destroy();

    noUiSlider.create(container, {
        start: [min, max],
        connect: true,
        step: (max - min) / 100 || 0.001,
        range: { min, max },
        tooltips: [
            { to: v => Number(v).toFixed(2) },
            { to: v => Number(v).toFixed(2) },
        ],
    });

    container.noUiSlider.on('update', (values) => {
        if (onUpdate) onUpdate(parseFloat(values[0]), parseFloat(values[1]));
    });

    return container.noUiSlider;
}
