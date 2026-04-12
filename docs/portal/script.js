/* RMath Documentation Portal — Interactivity */

document.addEventListener('DOMContentLoaded', () => {
    const sidebar = document.getElementById('sidebar');
    const searchInput = document.getElementById('moduleSearch');

    // 1. Mobile Menu
    const toggle = document.createElement('button');
    toggle.className = 'menu-toggle';
    toggle.setAttribute('aria-label', 'Toggle navigation');
    toggle.innerHTML = '<span></span><span></span><span></span>';
    document.body.appendChild(toggle);

    const overlay = document.createElement('div');
    overlay.className = 'overlay';
    document.body.appendChild(overlay);

    const closeSidebar = () => {
        sidebar.classList.remove('open');
        overlay.classList.remove('active');
    };

    toggle.addEventListener('click', () => {
        sidebar.classList.toggle('open');
        overlay.classList.toggle('active');
    });

    overlay.addEventListener('click', closeSidebar);

    // Close sidebar on nav link click (mobile)
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', () => {
            if (window.innerWidth <= 860) closeSidebar();
        });
    });

    // 2. Module search
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            const term = e.target.value.toLowerCase();
            document.querySelectorAll('#sidebarNav .nav-link').forEach(link => {
                const text = link.textContent.toLowerCase();
                link.style.display = text.includes(term) ? '' : 'none';
            });
            // Show group titles if any child is visible
            document.querySelectorAll('.nav-group').forEach(group => {
                const visible = group.querySelectorAll('.nav-link:not([style*="display: none"])');
                const title = group.querySelector('.nav-group-title');
                if (title) title.style.display = visible.length ? '' : 'none';
            });
        });
    }

    // 3. Copy buttons on code blocks
    document.querySelectorAll('.code-header').forEach(header => {
        const btn = document.createElement('button');
        btn.className = 'copy-btn';
        btn.textContent = 'Copy';
        btn.addEventListener('click', () => {
            const codeEl = header.nextElementSibling;
            if (!codeEl) return;
            const text = codeEl.innerText;
            navigator.clipboard.writeText(text).then(() => {
                btn.textContent = 'Copied!';
                setTimeout(() => btn.textContent = 'Copy', 1800);
            });
        });
        header.appendChild(btn);
    });

    // 4. Scroll spy for "On This Page" and anchor links
    const sections = document.querySelectorAll('section[id]');
    if (sections.length > 0) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const id = entry.target.getAttribute('id');
                    document.querySelectorAll('.nav-link').forEach(l => {
                        if (l.getAttribute('href') === `#${id}`) {
                            document.querySelectorAll('.nav-link[href^="#"]').forEach(x => x.classList.remove('active'));
                            l.classList.add('active');
                        }
                    });
                }
            });
        }, { threshold: 0.3, rootMargin: '-80px 0px -40% 0px' });

        sections.forEach(s => observer.observe(s));
    }

    // 5. Active page highlight in sidebar
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    document.querySelectorAll('.nav-link').forEach(link => {
        const href = link.getAttribute('href');
        if (href && !href.startsWith('#') && href === currentPage) {
            link.classList.add('active');
        }
    });
});
