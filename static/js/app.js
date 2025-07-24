// NCAA March Madness Predictor - Main JavaScript File

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initializeApp();
});

function initializeApp() {
    // Add smooth scrolling to all internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });

    // Add loading states to all buttons
    initializeButtonLoading();
    
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    // Initialize any page-specific functionality
    const currentPage = getCurrentPage();
    switch(currentPage) {
        case 'index':
            initializeDashboard();
            break;
        case 'predict':
            initializePrediction();
            break;
        case 'bracket':
            initializeBracket();
            break;
        case 'analytics':
            initializeAnalytics();
            break;
    }
}

function getCurrentPage() {
    const path = window.location.pathname;
    if (path === '/' || path === '/index') return 'index';
    if (path.includes('/predict')) return 'predict';
    if (path.includes('/bracket')) return 'bracket';
    if (path.includes('/analytics')) return 'analytics';
    return 'unknown';
}

function initializeButtonLoading() {
    // Add loading states to buttons with data-loading attribute
    document.querySelectorAll('button[data-loading]').forEach(button => {
        button.addEventListener('click', function() {
            if (!this.disabled) {
                addLoadingState(this);
            }
        });
    });
}

function addLoadingState(button) {
    const originalContent = button.innerHTML;
    const loadingText = button.getAttribute('data-loading') || 'Loading...';
    
    button.disabled = true;
    button.innerHTML = `<i class="fas fa-spinner fa-spin me-2"></i>${loadingText}`;
    
    // Store original content for restoration
    button.setAttribute('data-original-content', originalContent);
}

function removeLoadingState(button) {
    const originalContent = button.getAttribute('data-original-content');
    if (originalContent) {
        button.innerHTML = originalContent;
        button.disabled = false;
        button.removeAttribute('data-original-content');
    }
}

function initializeDashboard() {
    // Animate statistics cards on scroll
    const observeElements = document.querySelectorAll('.stat-card, .feature-card');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });
    
    observeElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'all 0.6s ease';
        observer.observe(el);
    });
    
    // Basketball rotation animation
    const basketball = document.querySelector('.basketball-animation i');
    if (basketball) {
        setInterval(() => {
            basketball.style.transform = 'rotate(360deg)';
            setTimeout(() => {
                basketball.style.transform = 'rotate(0deg)';
            }, 1000);
        }, 3000);
    }
}

function initializePrediction() {
    // Team selection enhancements
    const teamSelects = document.querySelectorAll('select[name*="team"]');
    
    teamSelects.forEach(select => {
        // Add search functionality to team selects
        if (select.options.length > 50) {
            makeSelectSearchable(select);
        }
        
        // Add change event listener
        select.addEventListener('change', function() {
            validateTeamSelection();
        });
    });
}

function makeSelectSearchable(select) {
    // Simple search functionality for team selection
    const wrapper = document.createElement('div');
    wrapper.className = 'searchable-select';
    
    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.className = 'form-control mb-2';
    searchInput.placeholder = 'Search teams...';
    
    select.parentNode.insertBefore(wrapper, select);
    wrapper.appendChild(searchInput);
    wrapper.appendChild(select);
    
    // Store original options
    const originalOptions = Array.from(select.options);
    
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        
        // Clear current options
        select.innerHTML = '<option value="">Select team...</option>';
        
        // Filter and add matching options
        originalOptions.slice(1).forEach(option => {
            if (option.text.toLowerCase().includes(searchTerm)) {
                select.appendChild(option.cloneNode(true));
            }
        });
    });
}

function validateTeamSelection() {
    const team1 = document.getElementById('team1');
    const team2 = document.getElementById('team2');
    const submitButton = document.querySelector('.predict-btn');
    
    if (team1 && team2 && submitButton) {
        const isValid = team1.value && team2.value && team1.value !== team2.value;
        submitButton.disabled = !isValid;
        
        if (team1.value === team2.value && team1.value) {
            showNotification('Please select different teams', 'warning');
        }
    }
}

function initializeBracket() {
    // Bracket visualization enhancements
    const regionCards = document.querySelectorAll('.region-card');
    
    regionCards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
            card.style.transition = 'all 0.6s ease';
        }, index * 150);
        
        // Add hover effects
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
}

function initializeAnalytics() {
    // Chart responsiveness
    window.addEventListener('resize', function() {
        // Redraw charts on window resize
        if (typeof Plotly !== 'undefined') {
            const chartDivs = document.querySelectorAll('[id*="Chart"]');
            chartDivs.forEach(div => {
                if (div.data) {
                    Plotly.Plots.resize(div);
                }
            });
        }
    });
    
    // Animate performance metrics
    const metrics = document.querySelectorAll('.performance-metric');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateMetricValue(entry.target);
            }
        });
    }, { threshold: 0.5 });
    
    metrics.forEach(metric => observer.observe(metric));
}

function animateMetricValue(metric) {
    const valueElement = metric.querySelector('.metric-value');
    if (!valueElement) return;
    
    const finalValue = valueElement.textContent;
    const isPercentage = finalValue.includes('%');
    const isDecimal = finalValue.includes('.');
    
    let numericValue = parseFloat(finalValue.replace(/[^\d.]/g, ''));
    if (isNaN(numericValue)) return;
    
    let currentValue = 0;
    const increment = numericValue / 50; // 50 steps
    
    const timer = setInterval(() => {
        currentValue += increment;
        
        if (currentValue >= numericValue) {
            currentValue = numericValue;
            clearInterval(timer);
        }
        
        let displayValue = isDecimal ? currentValue.toFixed(3) : Math.round(currentValue);
        if (isPercentage) displayValue += '%';
        if (finalValue.includes(',')) displayValue = displayValue.toLocaleString();
        
        valueElement.textContent = displayValue;
    }, 20);
}

// Utility Functions
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 100px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        <i class="fas fa-${getIconForType(type)} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

function getIconForType(type) {
    const icons = {
        'success': 'check-circle',
        'danger': 'exclamation-triangle',
        'warning': 'exclamation-circle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// API Helper Functions
async function makeApiCall(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        showNotification('An error occurred. Please try again.', 'danger');
        throw error;
    }
}

// Performance monitoring
function logPerformance(name, startTime) {
    const endTime = performance.now();
    const duration = endTime - startTime;
    console.log(`${name} took ${duration.toFixed(2)} milliseconds`);
}

// Progressive Web App functionality
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js')
            .then(function(registration) {
                console.log('ServiceWorker registration successful');
            })
            .catch(function(err) {
                console.log('ServiceWorker registration failed');
            });
    });
}

// Export functions for global access
window.NCAAPredictor = {
    showNotification,
    makeApiCall,
    addLoadingState,
    removeLoadingState,
    logPerformance
}; 