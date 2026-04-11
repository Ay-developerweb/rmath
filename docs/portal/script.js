/* RMath Documentation Portal - Premium Interactivity */

document.addEventListener('DOMContentLoaded', () => {
    const sidebar = document.querySelector('.sidebar');
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('section');
    const searchInput = document.getElementById('moduleSearch');
    const codeHeaders = document.querySelectorAll('.code-header');

    // 1. Mobile Menu Logic
    const toggleBtn = document.createElement('button');
    toggleBtn.className = 'menu-toggle';
    toggleBtn.innerHTML = '<span></span><span></span><span></span>';
    document.body.appendChild(toggleBtn);

    const overlay = document.createElement('div');
    overlay.className = 'sidebar-overlay';
    document.body.appendChild(overlay);

    toggleBtn.addEventListener('click', () => {
        sidebar.classList.toggle('open');
        overlay.classList.toggle('active');
    });

    overlay.addEventListener('click', () => {
        sidebar.classList.remove('open');
        overlay.classList.remove('active');
    });

    // 2. Module Search
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            const term = e.target.value.toLowerCase();
            document.querySelectorAll('#sidebarNav .nav-link').forEach(link => {
                const text = link.textContent.toLowerCase();
                link.style.display = text.includes(term) ? 'flex' : 'none';
            });
        });
    }

    // 3. Copy Code
    codeHeaders.forEach(header => {
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-btn';
        copyBtn.textContent = 'Copy';
        copyBtn.addEventListener('click', () => {
            const code = header.nextElementSibling.innerText;
            navigator.clipboard.writeText(code).then(() => {
                copyBtn.textContent = 'Copied!';
                setTimeout(() => copyBtn.textContent = 'Copy', 2000);
            });
        });
        header.appendChild(copyBtn);
    });

    // 4. Manual Click Active State (Instant Feedback)
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            if (link.getAttribute('href').startsWith('#')) {
                document.querySelectorAll('#onThisPage .nav-link').forEach(l => l.classList.remove('active'));
                link.classList.add('active');
            }
        });
    });

    // 5. Scroll Spy (On This Page)
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && entry.intersectionRatio >= 0.5) {
                const id = entry.target.getAttribute('id');
                const matchingLink = document.querySelector(`#onThisPage a[href="#${id}"]`);
                if (matchingLink) {
                    document.querySelectorAll('#onThisPage .nav-link').forEach(l => l.classList.remove('active'));
                    matchingLink.classList.add('active');
                }
            }
        });
    }, { threshold: [0.5] });

    sections.forEach(s => observer.observe(s));

    // Handle initial state
    const currentHash = window.location.hash;
    if (currentHash) {
        const matchingLink = document.querySelector(`#onThisPage a[href="${currentHash}"]`);
        if (matchingLink) matchingLink.classList.add('active');
    } else {
        const overviewLink = document.querySelector('#onThisPage a[href="#overview"]');
        if (overviewLink) overviewLink.classList.add('active');
    }
});
