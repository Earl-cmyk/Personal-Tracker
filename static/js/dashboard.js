/**
 * Personal Dashboard - Frontend Logic
 */

// Theme Management
class ThemeManager {
    constructor() {
        this.theme = localStorage.getItem('dashboard-theme') || 'night';
        this.init();
    }
    
    init() {
        this.applyTheme();
        this.setupListeners();
    }
    
    applyTheme() {
        document.documentElement.setAttribute('data-theme', this.theme);
        
        // Update icon
        const icon = document.querySelector('#themeToggle i');
        if (icon) {
            icon.className = this.theme === 'night' ? 'fas fa-sun' : 'fas fa-moon';
        }
    }
    
    setupListeners() {
        const toggleBtn = document.getElementById('themeToggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.toggleTheme());
        }
    }
    
    toggleTheme() {
        this.theme = this.theme === 'night' ? 'day' : 'night';
        localStorage.setItem('dashboard-theme', this.theme);
        this.applyTheme();
    }
}

// Metrics Manager
class MetricsManager {
    constructor() {
        this.baseUrl = '/api';
    }
    
    async fetchMetrics(connectionId) {
        try {
            const response = await fetch(`${this.baseUrl}/app/${connectionId}/metrics`);
            return await response.json();
        } catch (error) {
            console.error('Failed to fetch metrics:', error);
            return null;
        }
    }
    
    displayMetrics(container, metrics) {
        if (!metrics || !metrics.length) {
            container.innerHTML = '<div class="no-metrics">No metrics available</div>';
            return;
        }
        
        const html = metrics.map(metric => `
            <div class="metric-item">
                <div class="metric-label">${metric.label}</div>
                <div class="metric-value">${this.formatValue(metric.value)}</div>
                ${metric.change ? `
                    <div class="metric-change ${metric.change >= 0 ? 'positive' : 'negative'}">
                        ${metric.change >= 0 ? '↗' : '↘'} ${Math.abs(metric.change)}%
                    </div>
                ` : ''}
            </div>
        `).join('');
        
        container.innerHTML = html;
    }
    
    formatValue(value) {
        if (typeof value === 'number') {
            if (value >= 1000000) {
                return (value / 1000000).toFixed(1) + 'M';
            }
            if (value >= 1000) {
                return (value / 1000).toFixed(1) + 'K';
            }
            if (Number.isInteger(value)) {
                return value.toString();
            }
            return value.toFixed(2);
        }
        return value;
    }
}

// Notification Manager
class NotificationManager {
    constructor() {
        this.notifications = [];
        this.pollingInterval = null;
        this.pollingDelay = 300000; // 5 minutes
    }
    
    init() {
        this.loadNotifications();
        this.setupPolling();
        this.setupListeners();
    }
    
    async loadNotifications() {
        try {
            const response = await fetch('/notifications');
            this.notifications = await response.json();
            this.renderNotifications();
        } catch (error) {
            console.error('Failed to load notifications:', error);
        }
    }
    
    renderNotifications() {
        const container = document.querySelector('.notifications-list');
        if (!container) return;
        
        if (!this.notifications.length) {
            container.innerHTML = `
                <div class="notification-empty">
                    <i class="fas fa-inbox"></i>
                    <p>No notifications</p>
                </div>
            `;
            return;
        }
        
        const html = this.notifications.map(notification => `
            <div class="notification-item ${notification.priority} ${notification.read ? '' : 'unread'}" 
                 data-id="${notification.id}">
                <div class="notification-header">
                    <span class="notification-source">${notification.source || 'System'}</span>
                    <span class="notification-time">${this.formatTime(notification.created_at)}</span>
                </div>
                <div class="notification-content">
                    <strong>${notification.title}</strong>
                    <p>${notification.content}</p>
                </div>
            </div>
        `).join('');
        
        container.innerHTML = html;
        
        // Add click handlers
        container.querySelectorAll('.notification-item').forEach(item => {
            item.addEventListener('click', (e) => this.markAsRead(e));
        });
    }
    
    async markAsRead(event) {
        const item = event.currentTarget;
        const notificationId = item.dataset.id;
        
        try {
            await fetch(`/notifications/${notificationId}/read`, {
                method: 'POST'
            });
            
            item.classList.remove('unread');
        } catch (error) {
            console.error('Failed to mark as read:', error);
        }
    }
    
    async markAllAsRead() {
        try {
            await fetch('/notifications/read-all', {
                method: 'POST'
            });
            
            this.notifications.forEach(n => n.read = true);
            this.renderNotifications();
        } catch (error) {
            console.error('Failed to mark all as read:', error);
        }
    }
    
    formatTime(isoString) {
        const date = new Date(isoString);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);
        
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        if (diffDays < 7) return `${diffDays}d ago`;
        
        return date.toLocaleDateString();
    }
    
    setupPolling() {
        this.pollingInterval = setInterval(() => {
            this.loadNotifications();
        }, this.pollingDelay);
    }
    
    setupListeners() {
        const markAllBtn = document.querySelector('.btn-mark-all-read');
        if (markAllBtn) {
            markAllBtn.addEventListener('click', () => this.markAllAsRead());
        }
    }
    
    cleanup() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
        }
    }
}

// App Connection Manager
class AppConnectionManager {
    constructor() {
        this.metricsManager = new MetricsManager();
    }
    
    init() {
        this.setupSyncButtons();
        this.loadAllMetrics();
        // Edit functionality removed due to instability; keep delete handlers
        this.setupDeleteButtons();
        this.setupEditButtons();
    }

    setupEditButtons() {
        document.querySelectorAll('.btn-edit').forEach(button => {
            button.addEventListener('click', (e) => this.openEditModal(e));
        });

        // Modal close/cancel handlers
        const modalClose = document.getElementById('editModalClose');
        const editCancel = document.getElementById('editCancel');
        const editForm = document.getElementById('editForm');
        if (modalClose) modalClose.addEventListener('click', () => this.closeEditModal());
        if (editCancel) editCancel.addEventListener('click', () => this.closeEditModal());
        if (editForm) editForm.addEventListener('submit', (e) => this.submitEdit(e));
    }

    openEditModal(event) {
        const button = event.currentTarget;
        const connectionId = button.dataset.id;
        const card = button.closest('.app-card');

        const currentName = card.querySelector('.app-info h3')?.textContent || '';
        const currentUrl = card.querySelector('.profile-url small')?.textContent || '';

        const modal = document.getElementById('editModal');
        const inputName = document.getElementById('editAppName');
        const inputUrl = document.getElementById('editProfileUrl');

        if (!modal || !inputName || !inputUrl) return;

        modal.dataset.connectionId = connectionId;
        inputName.value = currentName.trim();
        inputUrl.value = currentUrl.replace(/^(\u2026|\.{3})$/, '').trim();

        modal.style.display = 'flex';
        modal.setAttribute('aria-hidden', 'false');
    }

    closeEditModal() {
        const modal = document.getElementById('editModal');
        if (!modal) return;
        modal.style.display = 'none';
        modal.removeAttribute('aria-hidden');
        delete modal.dataset.connectionId;
    }

    async submitEdit(event) {
        event.preventDefault();
        const modal = document.getElementById('editModal');
        if (!modal) return;
        const connectionId = modal.dataset.connectionId;
        const inputName = document.getElementById('editAppName');
        const inputUrl = document.getElementById('editProfileUrl');

        if (!connectionId) return;

        const payload = {
            app_name: inputName.value.trim(),
            profile_url: inputUrl.value.trim()
        };

        const saveBtn = document.getElementById('editSave');
        if (saveBtn) saveBtn.disabled = true;

        try {
            const resp = await fetch(`/connection/${connectionId}/update`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await resp.json();
            if (data.status === 'success') {
                // Update DOM
                const card = document.querySelector(`.app-card[data-id="${connectionId}"]`);
                if (card) {
                    const title = card.querySelector('.app-info h3');
                    if (title) title.textContent = data.app_name;

                    // profile url update
                    if (data.profile_url) {
                        let el = card.querySelector('.profile-url small');
                        if (!el) {
                            const footer = card.querySelector('.app-footer');
                            if (footer) {
                                const div = document.createElement('div');
                                div.className = 'profile-url';
                                div.innerHTML = `<small><i class="fas fa-link"></i> ${data.profile_url.length>30?data.profile_url.slice(0,30)+"...":data.profile_url}</small>`;
                                footer.appendChild(div);
                            }
                        } else {
                            el.innerHTML = `${data.profile_url.length>30?data.profile_url.slice(0,30)+"...":data.profile_url}`;
                        }
                    } else {
                        const el = card.querySelector('.profile-url');
                        if (el) el.remove();
                    }
                }

                showToast('Connection updated', 'success');
                this.closeEditModal();
            } else {
                showToast('Update failed: ' + (data.message || 'unknown'), 'error');
            }
        } catch (err) {
            console.error('Update error', err);
            showToast('Update failed', 'error');
        } finally {
            if (saveBtn) saveBtn.disabled = false;
        }
    }
    
    setupSyncButtons() {
        document.querySelectorAll('.btn-sync').forEach(button => {
            button.addEventListener('click', (e) => this.syncApp(e));
        });
    }

    setupDeleteButtons() {
        document.querySelectorAll('.btn-delete').forEach(button => {
            button.addEventListener('click', (e) => this.deleteConnection(e));
        });
    }

    async deleteConnection(event) {
        const button = event.currentTarget;
        const connectionId = button.dataset.id;
        const card = button.closest('.app-card');

        if (!confirm('Delete this connection? This cannot be undone.')) return;

        try {
            const resp = await fetch(`/connection/${connectionId}/delete`, { method: 'POST' });
            const data = await resp.json();
            if (data.status === 'success') {
                card.remove();
                showToast('Connection deleted', 'success');
            } else {
                showToast('Delete failed: ' + (data.message || 'unknown'), 'error');
            }
        } catch (err) {
            console.error('Delete error', err);
            showToast('Delete failed', 'error');
        }
    }
    
    async syncApp(event) {
        const button = event.currentTarget;
        const connectionId = button.dataset.id;
        const card = button.closest('.app-card');
        
        // Add loading state
        button.disabled = true;
        card.classList.add('syncing');
        
        try {
            const response = await fetch(`/sync/${connectionId}`);
            const data = await response.json();
            
            if (data.status === 'syncing') {
                // Wait a bit and reload metrics
                setTimeout(() => {
                    this.loadAppMetrics(connectionId);
                    button.disabled = false;
                    card.classList.remove('syncing');
                }, 2000);
            }
        } catch (error) {
            console.error('Sync failed:', error);
            button.disabled = false;
            card.classList.remove('syncing');
        }
    }
    
    async loadAllMetrics() {
        document.querySelectorAll('.app-card').forEach(async (card) => {
            const button = card.querySelector('.btn-sync');
            if (button) {
                const connectionId = button.dataset.id;
                await this.loadAppMetrics(connectionId);
            }
        });
    }
    
    async loadAppMetrics(connectionId) {
        const card = document.querySelector(`.app-card .btn-sync[data-id="${connectionId}"]`)?.closest('.app-card');
        if (!card) return;
        
        const metricsContainer = card.querySelector('.app-metrics');
        if (!metricsContainer) return;
        
        const metrics = await this.metricsManager.fetchMetrics(connectionId);
        if (metrics) {
            this.metricsManager.displayMetrics(metricsContainer, metrics);
        }
    }
}

// Dashboard Initialization
class Dashboard {
    constructor() {
        this.themeManager = new ThemeManager();
        this.notificationManager = new NotificationManager();
        this.appManager = new AppConnectionManager();
    }
    
    init() {
        this.themeManager.init();
        this.notificationManager.init();
        this.appManager.init();
        this.setupGlobalListeners();
        
        console.log('Dashboard initialized');
    }
    
    setupGlobalListeners() {
        // Refresh all apps button
        const refreshAllBtn = document.getElementById('refreshAll');
        if (refreshAllBtn) {
            refreshAllBtn.addEventListener('click', () => {
                document.querySelectorAll('.btn-sync').forEach(btn => btn.click());
            });
        }
        
        // Auto-refresh insights every hour
        setInterval(() => {
            this.refreshInsights();
        }, 3600000);
    }
    
    async refreshInsights() {
        try {
            const response = await fetch('/insights/daily');
            const insights = await response.json();
            
            // Update insight cards
            Object.keys(insights).forEach(category => {
                const card = document.querySelector(`.insight-card.${category} p`);
                if (card) {
                    card.textContent = insights[category];
                }
            });
        } catch (error) {
            console.error('Failed to refresh insights:', error);
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const dashboard = new Dashboard();
    dashboard.init();
});

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Export for debugging
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Dashboard, ThemeManager, MetricsManager };
}