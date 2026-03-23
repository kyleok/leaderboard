/* Cafe au LAIT — Theme Toggle (Light / Dark / System) */
(function () {
  'use strict';
  var KEY = 'lait-theme';
  var root = document.documentElement;

  function systemDark() {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  function apply(setting) {
    var dark = setting === 'dark' || (setting === 'system' && systemDark());
    root.classList.toggle('dark', dark);
    try { localStorage.setItem(KEY, setting); } catch (e) {}
  }

  function current() {
    try { return localStorage.getItem(KEY) || 'system'; } catch (e) { return 'system'; }
  }

  // Apply immediately (before paint)
  apply(current());

  // Listen for OS changes
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function () {
      if (current() === 'system') apply('system');
    });
  }

  // Inject toggle button — call after DOM ready
  function injectToggle(container) {
    if (!container || container.querySelector('.lait-theme-toggle')) return;
    var btn = document.createElement('button');
    btn.className = 'lait-theme-toggle';
    btn.title = 'Toggle theme';
    btn.setAttribute('aria-label', 'Toggle theme');
    btn.style.cssText = 'background:none;border:1px solid var(--border);border-radius:8px;padding:6px 8px;cursor:pointer;color:var(--text-muted);display:inline-flex;align-items:center;transition:border-color .2s,color .2s;';
    var SUN = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/></svg>';
    var MOON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.985 12.486a9 9 0 1 1-9.473-9.472c.405-.022.617.46.402.803a6 6 0 0 0 8.268 8.268c.344-.215.825-.004.803.401"/></svg>';

    function update() {
      var s = current();
      var dark = s === 'dark' || (s === 'system' && systemDark());
      btn.innerHTML = dark ? MOON : SUN;
    }
    update();

    btn.addEventListener('click', function () {
      var order = ['light', 'dark', 'system'];
      var next = order[(order.indexOf(current()) + 1) % 3];
      apply(next);
      update();
      btn.title = 'Theme: ' + next;
    });

    btn.addEventListener('mouseenter', function () { btn.style.borderColor = 'var(--accent)'; btn.style.color = 'var(--text)'; });
    btn.addEventListener('mouseleave', function () { btn.style.borderColor = 'var(--border)'; btn.style.color = 'var(--text-muted)'; });

    container.appendChild(btn);
  }

  // Expose globally
  window.__laitTheme = { apply: apply, current: current, injectToggle: injectToggle };
})();
