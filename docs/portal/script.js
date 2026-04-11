/**
 * RMath Docs — Mobile Responsive JS
 * Handles: sidebar toggle, overlay, copy buttons, active nav links
 */
(function () {
    'use strict';

    /* ── Sidebar Toggle ───────────────────────────────────────── */
    const sidebar    = document.querySelector('aside.sidebar');
    const overlay    = document.querySelector('.sidebar-overlay');
    const menuToggle = document.querySelector('.menu-toggle');
    const body       = document.body;

    function openSidebar() {
        sidebar?.classList.add('open');
        overlay?.classList.add('active');
        menuToggle?.classList.add('open');
        menuToggle?.setAttribute('aria-expanded', 'true');
        body.classList.add('sidebar-open');
    }

    function closeSidebar() {
        sidebar?.classList.remove('open');
        overlay?.classList.remove('active');
        menuToggle?.classList.remove('open');
        menuToggle?.setAttribute('aria-expanded', 'false');
        body.classList.remove('sidebar-open');
    }

    menuToggle?.addEventListener('click', () => {
        sidebar?.classList.contains('open') ? closeSidebar() : openSidebar();
    });

    overlay?.addEventListener('click', closeSidebar);

    /* Close sidebar on nav link click (mobile UX) */
    sidebar?.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', () => {
            if (window.innerWidth <= 1024) closeSidebar();
        });
    });

    /* Close on Escape key */
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') closeSidebar();
    });

    /* Restore on resize to desktop */
    window.addEventListener('resize', () => {
        if (window.innerWidth > 1024) closeSidebar();
    });

    /* ── Copy Buttons ─────────────────────────────────────────── */
    document.querySelectorAll('.copy-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const pre = btn.closest('.code-window')?.querySelector('pre');
            if (!pre) return;

            const text = pre.innerText || pre.textContent;

            try {
                await navigator.clipboard.writeText(text);
                btn.textContent = '✓ Copied';
                btn.classList.add('copied');
                setTimeout(() => {
                    btn.textContent = 'Copy';
                    btn.classList.remove('copied');
                }, 2000);
            } catch {
                /* Fallback for older browsers */
                const ta = document.createElement('textarea');
                ta.value = text;
                ta.style.cssText = 'position:fixed;opacity:0;';
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                ta.remove();
                btn.textContent = '✓ Copied';
                btn.classList.add('copied');
                setTimeout(() => {
                    btn.textContent = 'Copy';
                    btn.classList.remove('copied');
                }, 2000);
            }
        });
    });

    /* ── Active Nav Link on Scroll ───────────────────────────── */
    const sections  = document.querySelectorAll('section[id], div[id]');
    const navLinks  = document.querySelectorAll('aside.sidebar .nav-link[href^="#"]');

    if (sections.length && navLinks.length) {
        const observer = new IntersectionObserver(entries => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    navLinks.forEach(link => {
                        link.classList.toggle(
                            'active',
                            link.getAttribute('href') === `#${entry.target.id}`
                        );
                    });
                }
            });
        }, { rootMargin: '-20% 0px -70% 0px' });

        sections.forEach(s => observer.observe(s));
    }

    /* ── Accessibility: ARIA on sidebar ──────────────────────── */
    sidebar?.setAttribute('role', 'navigation');
    sidebar?.setAttribute('aria-label', 'Documentation navigation');
    menuToggle?.setAttribute('aria-expanded', 'false');
    menuToggle?.setAttribute('aria-controls', 'sidebar');
    menuToggle?.setAttribute('aria-label', 'Toggle navigation menu');
    if (sidebar) sidebar.id = 'sidebar';
})();
